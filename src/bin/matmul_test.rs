use candle_core::{DType, Device, IndexOp, Tensor};
use std::fs;

fn main() -> anyhow::Result<()> {
    candle_core::cuda_backend::set_gemm_reduced_precision_bf16(true);

    let device = Device::cuda_if_available(0)?;

    // Load test data (saved as F32)
    let x_bin = fs::read("debug_matmul_x.bin")?;
    let x_f32: Vec<f32> = x_bin.chunks(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    let x = Tensor::from_vec(x_f32, (39, 3072), &device)?.to_dtype(DType::BF16)?.unsqueeze(0)?;

    let w_bin = fs::read("debug_matmul_w.bin")?;
    let w_f32: Vec<f32> = w_bin.chunks(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
    let w = Tensor::from_vec(w_f32, (3072, 4096), &device)?.to_dtype(DType::BF16)?;

    // Compute matmul - candle requires matching dims for batch matmul
    let w_3d = w.unsqueeze(0)?;
    let result = x.matmul(&w_3d)?;

    // Load PyTorch result
    let r_bin = fs::read("debug_matmul_result.bin")?;
    let r_py: Vec<f32> = r_bin.chunks(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();

    // Compare
    let r_candle = result.to_dtype(DType::F32)?.squeeze(0)?; // [39, 4096]
    let r_candle_flat: Vec<f32> = r_candle.flatten_all()?.to_vec1()?;

    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f64 = 0.0;
    let mut count = 0usize;
    for i in 0..r_py.len().min(r_candle_flat.len()) {
        let d = (r_py[i] - r_candle_flat[i]).abs();
        if d > max_diff { max_diff = d; }
        sum_diff += d as f64;
        count += 1;
    }
    let mean_diff = sum_diff / count as f64;

    // Print first5 for row 0 and row 38
    let r0: Vec<f32> = r_candle.i((0, ..5))?.to_vec1()?;
    let r38: Vec<f32> = r_candle.i((38, ..5))?.to_vec1()?;

    println!("Candle result[0,:5]: {:?}", r0);
    println!("PyTorch result[0,:5]: {:?}", &r_py[..5]);
    println!("Candle result[38,:5]: {:?}", r38);
    let offset = 38 * 4096;
    println!("PyTorch result[38,:5]: {:?}", &r_py[offset..offset+5]);
    println!("Max diff: {:.6}", max_diff);
    println!("Mean diff: {:.6}", mean_diff);

    Ok(())
}
