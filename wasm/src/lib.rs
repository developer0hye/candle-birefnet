//! WASM bindings for BiRefNet_lite background removal.
//!
//! Exports two functions via wasm-bindgen:
//! - `load_model(weights)` — load SafeTensors weights (FP16 or INT8) into memory
//! - `segment(pixels, width, height)` — run segmentation, return alpha mask

use std::collections::HashMap;
use std::sync::Mutex;

use candle_birefnet::BiRefNet;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use wasm_bindgen::prelude::*;

static MODEL: Mutex<Option<BiRefNet>> = Mutex::new(None);

/// ImageNet normalization constants.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Model input resolution.
/// 1024 is the training resolution but requires ~4 GB memory in WASM.
/// 512 is a practical compromise: good quality, fits in WASM memory (~1 GB).
const MODEL_SIZE: usize = 512;

/// Load SafeTensors bytes, dequantize INT8 tensors, and build BiRefNet_lite.
///
/// INT8 tensors are identified by companion `{name}.__scale` and `{name}.__zero_point`
/// tensors. Dequantization: `weight_fp32 = (weight_int8 - zero_point) * scale`
///
/// FP16 tensors are converted to FP32. Regular FP32 tensors pass through unchanged.
fn load_and_build_model(data: &[u8], device: &Device) -> Result<BiRefNet, String> {
    let tensors = safetensors::SafeTensors::deserialize(data)
        .map_err(|e| format!("Failed to deserialize safetensors: {e}"))?;

    // First pass: load all tensors into a map
    let mut raw: HashMap<String, Tensor> = HashMap::new();
    for (name, view) in tensors.tensors() {
        let dtype = match view.dtype() {
            safetensors::Dtype::F32 => DType::F32,
            safetensors::Dtype::F16 => DType::F16,
            safetensors::Dtype::I8 => DType::I64, // Load I8 as raw bytes, handle below
            safetensors::Dtype::I64 => DType::I64,
            safetensors::Dtype::U32 => DType::U32,
            _ => continue,
        };

        // For I8 tensors, load as raw i8 values manually
        if view.dtype() == safetensors::Dtype::I8 {
            let i8_data: &[u8] = view.data();
            let f32_data: Vec<f32> = i8_data.iter().map(|&b| (b as i8) as f32).collect();
            let tensor = Tensor::from_vec(f32_data, view.shape(), device)
                .map_err(|e| format!("Failed to load I8 tensor {name}: {e}"))?;
            raw.insert(name.to_string(), tensor);
        } else {
            let tensor = Tensor::from_raw_buffer(view.data(), dtype, view.shape(), device)
                .map_err(|e| format!("Failed to load tensor {name}: {e}"))?;
            raw.insert(name.to_string(), tensor);
        }
    }

    // Second pass: dequantize INT8 weights, convert FP16 to FP32
    let varmap = VarMap::new();
    {
        let mut data_map = varmap.data().lock().unwrap();

        // Collect weight keys (skip __scale and __zero_point)
        let weight_keys: Vec<String> = raw
            .keys()
            .filter(|k| !k.ends_with(".__scale") && !k.ends_with(".__zero_point"))
            .cloned()
            .collect();

        for key in weight_keys {
            let tensor = &raw[&key];
            let scale_key = format!("{key}.__scale");
            let zp_key = format!("{key}.__zero_point");

            let final_tensor = if raw.contains_key(&scale_key) {
                // INT8 weight: dequantize = (int8_as_f32 - zero_point) * scale
                let scale = raw[&scale_key].to_dtype(DType::F32)
                    .map_err(|e| format!("Scale conversion error for {key}: {e}"))?;
                let zero_point = &raw[&zp_key];
                let zp_f32 = zero_point.to_dtype(DType::F32)
                    .map_err(|e| format!("ZP conversion error for {key}: {e}"))?;

                let w_f32 = tensor.to_dtype(DType::F32)
                    .map_err(|e| format!("Weight conversion error for {key}: {e}"))?;

                // Broadcast scale/zp: shape [out_channels] → [out_channels, 1, 1, ...]
                let w_dims = w_f32.dims().len();
                let dequantized = if w_dims > 1 {
                    let mut shape = vec![scale.dims()[0]];
                    for _ in 1..w_dims {
                        shape.push(1);
                    }
                    let scale_b = scale.reshape(&shape[..])
                        .map_err(|e| format!("Scale reshape error for {key}: {e}"))?;
                    let zp_b = zp_f32.reshape(&shape[..])
                        .map_err(|e| format!("ZP reshape error for {key}: {e}"))?;
                    w_f32.broadcast_sub(&zp_b)
                        .and_then(|t| t.broadcast_mul(&scale_b))
                        .map_err(|e| format!("Dequantize error for {key}: {e}"))?
                } else {
                    w_f32.sub(&zp_f32)
                        .and_then(|t| t.mul(&scale))
                        .map_err(|e| format!("Dequantize error for {key}: {e}"))?
                };

                dequantized
            } else {
                // FP16/FP32: convert to FP32
                tensor.to_dtype(DType::F32)
                    .map_err(|e| format!("Dtype conversion error for {key}: {e}"))?
            };

            data_map.insert(
                key,
                Var::from_tensor(&final_tensor)
                    .map_err(|e| format!("Var creation error: {e}"))?,
            );
        }
    }

    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = BiRefNet::new_lite(vb)
        .map_err(|e| format!("Failed to build model: {e}"))?;
    Ok(model)
}

/// Load BiRefNet_lite model from SafeTensors bytes.
/// Supports both FP16 and INT8 (with dequantization) weight formats.
#[wasm_bindgen]
pub fn load_model(weights: &[u8]) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    let device = Device::Cpu;
    let model = load_and_build_model(weights, &device)
        .map_err(|e| JsValue::from_str(&e))?;

    let mut guard = MODEL
        .lock()
        .map_err(|e| JsValue::from_str(&format!("Lock error: {e}")))?;
    *guard = Some(model);
    Ok(())
}

/// Run background segmentation on RGBA pixels.
///
/// Input: RGBA pixel buffer (width * height * 4 bytes)
/// Output: Alpha mask (width * height bytes, 0=background, 255=foreground)
#[wasm_bindgen]
pub fn segment(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
    let guard = MODEL
        .lock()
        .map_err(|e| JsValue::from_str(&format!("Lock error: {e}")))?;
    let model = guard
        .as_ref()
        .ok_or_else(|| JsValue::from_str("Model not loaded. Call load_model() first."))?;

    let device = Device::Cpu;
    let w = width as usize;
    let h = height as usize;

    // Preprocess: RGBA → RGB normalized tensor [1, 3, MODEL_SIZE, MODEL_SIZE]
    let input = preprocess(pixels, w, h, &device)
        .map_err(|e| JsValue::from_str(&format!("Preprocess error: {e}")))?;

    // Inference
    let outputs = model
        .forward(&input)
        .map_err(|e| JsValue::from_str(&format!("Inference error: {e}")))?;

    // Postprocess: sigmoid → resize → u8 mask
    let mask = postprocess(&outputs[0], w, h)
        .map_err(|e| JsValue::from_str(&format!("Postprocess error: {e}")))?;

    Ok(mask)
}

/// Convert RGBA pixels to ImageNet-normalized RGB tensor at model resolution.
fn preprocess(pixels: &[u8], w: usize, h: usize, device: &Device) -> candle_core::Result<Tensor> {
    let pixel_count = w * h;
    let mut rgb = vec![0f32; 3 * pixel_count];
    for i in 0..pixel_count {
        let rgba_idx = i * 4;
        rgb[i] = pixels[rgba_idx] as f32 / 255.0;
        rgb[pixel_count + i] = pixels[rgba_idx + 1] as f32 / 255.0;
        rgb[2 * pixel_count + i] = pixels[rgba_idx + 2] as f32 / 255.0;
    }

    let tensor = Tensor::from_vec(rgb, (1, 3, h, w), device)?;

    let mean = Tensor::from_slice(&MEAN, (1, 3, 1, 1), device)?;
    let std = Tensor::from_slice(&STD, (1, 3, 1, 1), device)?;
    let normalized = tensor.broadcast_sub(&mean)?.broadcast_div(&std)?;

    let resized = normalized.upsample_bilinear2d(MODEL_SIZE, MODEL_SIZE, true)?;

    Ok(resized)
}

/// Convert model output logits to u8 alpha mask at original resolution.
fn postprocess(logits: &Tensor, orig_w: usize, orig_h: usize) -> candle_core::Result<Vec<u8>> {
    let probs = candle_nn::ops::sigmoid(logits)?;
    let resized = probs.upsample_bilinear2d(orig_h, orig_w, true)?;

    let flat = resized.squeeze(0)?.squeeze(0)?.flatten_all()?;
    let values = flat.to_vec1::<f32>()?;

    let mask: Vec<u8> = values
        .iter()
        .map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8)
        .collect();

    Ok(mask)
}
