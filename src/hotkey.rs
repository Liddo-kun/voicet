// Global keyboard listener via rdev: hotkey toggle + Ctrl+C detection.

use anyhow::{bail, Result};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub const STATE_READY: u8 = 0;
pub const STATE_ACTIVE: u8 = 1;
pub const STATE_PAUSED: u8 = 2;

pub fn parse_hotkey(s: &str) -> Result<rdev::Key> {
    match s {
        "F1" => Ok(rdev::Key::F1),
        "F2" => Ok(rdev::Key::F2),
        "F3" => Ok(rdev::Key::F3),
        "F4" => Ok(rdev::Key::F4),
        "F5" => Ok(rdev::Key::F5),
        "F6" => Ok(rdev::Key::F6),
        "F7" => Ok(rdev::Key::F7),
        "F8" => Ok(rdev::Key::F8),
        "F9" => Ok(rdev::Key::F9),
        "F10" => Ok(rdev::Key::F10),
        "F11" => Ok(rdev::Key::F11),
        "F12" => Ok(rdev::Key::F12),
        "ScrollLock" => Ok(rdev::Key::ScrollLock),
        "Pause" => Ok(rdev::Key::Pause),
        "PrintScreen" => Ok(rdev::Key::PrintScreen),
        _ => bail!(
            "Unknown hotkey '{}'. Valid keys: F1-F12, ScrollLock, Pause, PrintScreen",
            s
        ),
    }
}

/// Spawn rdev listener thread. Always handles Ctrl+C → sets `running` to false.
/// If `hotkey` is Some, also toggles `state` on that key with 200ms debounce.
pub fn spawn_listener(
    running: Arc<AtomicBool>,
    hotkey: Option<rdev::Key>,
    state: Option<Arc<AtomicU8>>,
) {
    std::thread::spawn(move || {
        let mut ctrl_held = false;
        let mut last_press = Instant::now();
        let debounce = std::time::Duration::from_millis(200);

        if let Err(e) = rdev::listen(move |event| match event.event_type {
            rdev::EventType::KeyPress(k) => {
                if matches!(k, rdev::Key::ControlLeft | rdev::Key::ControlRight) {
                    ctrl_held = true;
                }
                if k == rdev::Key::KeyC && ctrl_held {
                    running.store(false, Ordering::SeqCst);
                }
                if let (Some(hk), Some(ref st)) = (hotkey, &state) {
                    if k == hk && last_press.elapsed() >= debounce {
                        last_press = Instant::now();
                        if st.load(Ordering::SeqCst) == STATE_ACTIVE {
                            st.store(STATE_PAUSED, Ordering::SeqCst);
                            eprintln!("[paused]");
                        } else {
                            st.store(STATE_ACTIVE, Ordering::SeqCst);
                            eprintln!("[recording]");
                        }
                    }
                }
            }
            rdev::EventType::KeyRelease(k) => {
                if matches!(k, rdev::Key::ControlLeft | rdev::Key::ControlRight) {
                    ctrl_held = false;
                }
            }
            _ => {}
        }) {
            eprintln!("Keyboard listener error: {:?}", e);
        }
    });
}
