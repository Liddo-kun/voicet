// Shared settings infrastructure for GUI and CLI configuration.
//
// SharedSettings holds all adjustable parameters as atomics, shared between
// the inference thread, hotkey thread, and (future) tray/settings UI threads.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::hotkey;

/// Atomic f32 via bit reinterpretation (no std AtomicF32).
pub struct AtomicF32(AtomicU32);

impl AtomicF32 {
    pub fn new(v: f32) -> Self {
        Self(AtomicU32::new(v.to_bits()))
    }
    pub fn load(&self, order: Ordering) -> f32 {
        f32::from_bits(self.0.load(order))
    }
    pub fn store(&self, v: f32, order: Ordering) {
        self.0.store(v.to_bits(), order);
    }
}

/// All GUI-adjustable settings as atomic types.
pub struct SharedSettings {
    pub silence_threshold: AtomicF32,
    pub silence_chunks: AtomicUsize,
    pub min_speech_chunks: AtomicUsize,
    pub rms_ema_alpha: AtomicF32,
    pub delay_tokens: AtomicUsize,
    pub hotkey: Mutex<Option<rdev::Key>>,
    pub type_mode: AtomicBool,
    pub state: AtomicU8,
}

impl SharedSettings {
    /// Construct from resolved IniValues. `silence_chunks` and `initial_state`
    /// are passed separately because they're computed from other values.
    pub fn new(vals: IniValues, silence_chunks: usize, initial_state: u8) -> Self {
        Self {
            silence_threshold: AtomicF32::new(vals.silence_threshold),
            silence_chunks: AtomicUsize::new(silence_chunks),
            min_speech_chunks: AtomicUsize::new(vals.min_speech_chunks),
            rms_ema_alpha: AtomicF32::new(vals.rms_ema_alpha),
            delay_tokens: AtomicUsize::new(vals.delay),
            hotkey: Mutex::new(vals.hotkey),
            type_mode: AtomicBool::new(vals.type_mode),
            state: AtomicU8::new(initial_state),
        }
    }
}

/// Intermediate non-atomic settings for INI loading and CLI merging.
pub struct IniValues {
    pub delay: usize,
    pub silence_threshold: f32,
    pub silence_chunks: Option<usize>,
    pub min_speech_chunks: usize,
    pub rms_ema_alpha: f32,
    pub hotkey: Option<rdev::Key>,
    pub type_mode: bool,
}

impl Default for IniValues {
    fn default() -> Self {
        Self {
            delay: 4,
            silence_threshold: 0.006,
            silence_chunks: None,
            min_speech_chunks: 12,
            rms_ema_alpha: 0.3,
            hotkey: None,
            type_mode: true,
        }
    }
}

fn parse_ini(contents: &str) -> IniValues {
    let mut vals = IniValues::default();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim();
            let value = value.trim();
            match key {
                "delay" => { if let Ok(v) = value.parse() { vals.delay = v; } }
                "silence_threshold" => { if let Ok(v) = value.parse() { vals.silence_threshold = v; } }
                "silence_chunks" => { if let Ok(v) = value.parse() { vals.silence_chunks = Some(v); } }
                "min_speech_chunks" => { if let Ok(v) = value.parse() { vals.min_speech_chunks = v; } }
                "rms_ema_alpha" => { if let Ok(v) = value.parse() { vals.rms_ema_alpha = v; } }
                "hotkey" => {
                    if value.eq_ignore_ascii_case("none") || value.is_empty() {
                        vals.hotkey = None;
                    } else if let Ok(k) = hotkey::parse_hotkey(value) {
                        vals.hotkey = Some(k);
                    }
                }
                "output_mode" => {
                    vals.type_mode = value == "type";
                }
                _ => {}
            }
        }
    }
    vals
}

/// Load settings from an INI file. Missing keys use defaults. Missing file = all defaults.
pub fn load_ini(path: &Path) -> IniValues {
    match std::fs::read_to_string(path) {
        Ok(contents) => parse_ini(&contents),
        Err(_) => IniValues::default(),
    }
}

/// Save current settings to an INI file.
pub fn save_settings(path: &Path, settings: &SharedSettings) {
    let hotkey_str = match *settings.hotkey.lock().unwrap() {
        Some(key) => hotkey::key_to_string(key),
        None => "none".to_string(),
    };
    let output_mode = if settings.type_mode.load(Ordering::Relaxed) { "type" } else { "none" };

    let content = format!(
        "delay={}\nsilence_threshold={}\nsilence_chunks={}\nmin_speech_chunks={}\nrms_ema_alpha={}\nhotkey={}\noutput_mode={}\n",
        settings.delay_tokens.load(Ordering::Relaxed),
        settings.silence_threshold.load(Ordering::Relaxed),
        settings.silence_chunks.load(Ordering::Relaxed),
        settings.min_speech_chunks.load(Ordering::Relaxed),
        settings.rms_ema_alpha.load(Ordering::Relaxed),
        hotkey_str,
        output_mode,
    );

    let _ = std::fs::write(path, content);
}

/// Path to settings.ini next to the executable.
pub fn settings_path() -> std::path::PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("settings.ini")
}

/// Reload settings from INI file into shared atomics. Skips on read error.
pub fn reload_from_file(settings: &SharedSettings, path: &Path, hotkey_thread_id: &AtomicU32) {
    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return,
    };
    let vals = parse_ini(&contents);

    settings.delay_tokens.store(vals.delay, Ordering::Relaxed);
    settings.silence_threshold.store(vals.silence_threshold, Ordering::Relaxed);
    if let Some(sc) = vals.silence_chunks {
        settings.silence_chunks.store(sc, Ordering::Relaxed);
    }
    settings.min_speech_chunks.store(vals.min_speech_chunks, Ordering::Relaxed);
    settings.rms_ema_alpha.store(vals.rms_ema_alpha, Ordering::Relaxed);
    settings.type_mode.store(vals.type_mode, Ordering::Relaxed);

    let mut current = settings.hotkey.lock().unwrap();
    if *current != vals.hotkey {
        *current = vals.hotkey;
        drop(current);
        hotkey::change_hotkey(hotkey_thread_id, vals.hotkey);
    }
}
