# INT8 Quantization Guide for BiRefNet_lite

This document describes the full INT8 post-training quantization (PTQ) pipeline for BiRefNet_lite, from calibration data preparation through weight generation and Rust-side dequantization.

## Overview

| Variant | Format | Size | IoU vs FP32 |
|---------|--------|------|-------------|
| Original | FP32 | 178 MB | 1.0000 (baseline) |
| FP16 | FP16 | 85 MB | ~1.0000 |
| **INT8 (PTQ)** | **INT8 + FP16 mixed** | **43 MB** | **0.9986** |

INT8 quantization reduces model weights to 25% of the original size while maintaining IoU > 0.998 against FP32 inference.

## Quantization Method

We use **PyTorch Post-Training Static Quantization** (`torch.ao.quantization`) with:

- **Weight quantization**: Per-channel symmetric INT8 (`PerChannelMinMaxObserver`)
- **Activation observer**: `HistogramObserver` — collects activation distributions during calibration and finds optimal scale by minimizing quantization error across the histogram (more accurate than simple min/max)
- **Calibration dataset**: [DUTS-TE](https://saliencydetection.net/duts/) (5,019 salient object images), 500 images sampled with seed 42
- **Mixed precision**: Layers that PyTorch cannot quantize (LayerNorm, custom modules) remain in FP16

### Why not simple min/max quantization?

Simple per-channel min/max quantization (without calibration) fails catastrophically on some images:

| Image | Simple MinMax IoU | PyTorch PTQ IoU |
|-------|-------------------|-----------------|
| Helicopter | 0.001 | **0.9986** |
| Windmill | 0.967 | **0.9980** |

The `HistogramObserver` uses calibration data to build activation histograms and find optimal quantization boundaries that minimize overall error, rather than being distorted by outlier values.

## Step-by-Step Reproduction

### Prerequisites

```bash
pip install torch torchvision safetensors huggingface_hub pillow

# Clone BiRefNet (needed for model architecture)
git clone https://github.com/ZhengPeng7/BiRefNet.git ~/Documents/Projects/BiRefNet
```

### Step 1: Download Calibration Dataset

```bash
# Download DUTS-TE (133 MB, 5,019 images)
curl -L -o /tmp/duts-te.zip "http://saliencydetection.net/duts/download/DUTS-TE.zip"
unzip -q /tmp/duts-te.zip -d /tmp/duts-te
# Images are at: /tmp/duts-te/DUTS-TE/DUTS-TE-Image/
```

DUTS-TE is a standard salient object detection benchmark from ImageNet. It was NOT used to train BiRefNet, making it suitable for calibration (avoids data leakage).

### Step 2: Run Quantization

```bash
python scripts/quantize_int8_ptq.py \
  --dataset-dir /tmp/duts-te/DUTS-TE/DUTS-TE-Image \
  --num-calibration 500 \
  --output weights/birefnet_int8.safetensors
```

This takes ~30 minutes on CPU (500 forward passes through BiRefNet_lite at 512x512).

**What happens internally:**

1. Load BiRefNet_lite with Swin-V1-Tiny backbone (FP32)
2. Attach `HistogramObserver` (activations) and `PerChannelMinMaxObserver` (weights) to all quantizable layers
3. Run 500 calibration images through the model — observers collect statistics
4. `torch.ao.quantization.convert()` replaces FP32 layers with quantized versions using optimal scale/zero_point
5. Extract INT8 weights + per-channel scale/zero_point from quantized model
6. Save as SafeTensors

### Step 3: Verify Quality

```bash
python scripts/compare_int8_quality.py
```

Output:
```
Helicopter: max_diff=0.203725, mean_diff=0.000375, IoU=0.9986
Windmill:   max_diff=0.486941, mean_diff=0.000568, IoU=0.9980
```

## SafeTensors Format

The INT8 SafeTensors file contains three types of tensors:

### 1. Quantized weights (INT8)

```
"bb.patch_embed.proj.weight"          → INT8 tensor [96, 3, 4, 4]
"bb.patch_embed.proj.weight.__scale"  → FP32 tensor [96]      (per output channel)
"bb.patch_embed.proj.weight.__zero_point" → INT8 tensor [96]   (per output channel)
```

### 2. Non-quantized parameters (FP16)

Biases, LayerNorm weights, relative position bias tables, and other 1D/small tensors:

```
"bb.patch_embed.proj.bias"            → FP16 tensor [96]
"bb.layers.0.blocks.0.norm1.weight"   → FP16 tensor [96]
```

### Key convention

- If a tensor `{name}` has companion tensors `{name}.__scale` and `{name}.__zero_point`, it is INT8 quantized
- Otherwise, it is stored directly (FP16 or FP32)

## Rust Dequantization (candle)

At model loading time, INT8 weights are dequantized to FP32 before building the model. This happens in the WASM loader (`wasm/src/lib.rs`), but the same logic applies to any candle application.

### Dequantization formula

```
weight_fp32 = (weight_int8_as_f32 - zero_point_as_f32) * scale
```

Per-channel: scale and zero_point have shape `[out_channels]` and are broadcast across the remaining dimensions.

### Rust implementation

```rust
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;

fn load_int8_safetensors(data: &[u8], device: &Device) -> candle_core::Result<VarBuilder<'_>> {
    let tensors = safetensors::SafeTensors::deserialize(data)?;

    // Load all tensors
    let mut raw: HashMap<String, Tensor> = HashMap::new();
    for (name, view) in tensors.tensors() {
        if view.dtype() == safetensors::Dtype::I8 {
            // Load INT8 as f32 values
            let i8_data: &[u8] = view.data();
            let f32_data: Vec<f32> = i8_data.iter().map(|&b| (b as i8) as f32).collect();
            raw.insert(name.to_string(),
                Tensor::from_vec(f32_data, view.shape(), device)?);
        } else {
            let dtype = match view.dtype() {
                safetensors::Dtype::F32 => DType::F32,
                safetensors::Dtype::F16 => DType::F16,
                _ => continue,
            };
            raw.insert(name.to_string(),
                Tensor::from_raw_buffer(view.data(), dtype, view.shape(), device)?);
        }
    }

    // Dequantize and populate VarMap
    let varmap = VarMap::new();
    let mut data_map = varmap.data().lock().unwrap();

    for key in raw.keys().filter(|k| !k.ends_with(".__scale") && !k.ends_with(".__zero_point")) {
        let tensor = &raw[key];
        let scale_key = format!("{key}.__scale");
        let zp_key = format!("{key}.__zero_point");

        let final_tensor = if raw.contains_key(&scale_key) {
            // Dequantize: (int8 - zero_point) * scale
            let scale = raw[&scale_key].to_dtype(DType::F32)?;
            let zp = raw[&zp_key].to_dtype(DType::F32)?;
            let w = tensor.to_dtype(DType::F32)?;

            // Broadcast scale/zp from [out_channels] to weight shape
            let mut shape = vec![scale.dims()[0]];
            for _ in 1..w.dims().len() {
                shape.push(1);
            }
            let scale_b = scale.reshape(&shape[..])?;
            let zp_b = zp.reshape(&shape[..])?;

            w.broadcast_sub(&zp_b)?.broadcast_mul(&scale_b)?
        } else {
            tensor.to_dtype(DType::F32)?
        };

        data_map.insert(key.clone(), Var::from_tensor(&final_tensor)?);
    }
    drop(data_map);

    Ok(VarBuilder::from_varmap(&varmap, DType::F32, device))
}
```

### Usage

```rust
let weights = std::fs::read("birefnet_int8.safetensors")?;
let device = &Device::Cpu;
let vb = load_int8_safetensors(&weights, device)?;
let model = BiRefNet::new_lite(vb)?;
// model is ready — weights are FP32 after dequantization
```

### Memory note

Dequantization converts INT8 → FP32 at load time, so **runtime memory usage is the same as FP32**. The benefit is purely in **download/storage size** (43 MB vs 85 MB FP16 vs 178 MB FP32).

## File Reference

| File | Description |
|------|-------------|
| `scripts/quantize_int8_ptq.py` | PyTorch PTQ quantization with DUTS-TE calibration |
| `scripts/compare_int8_quality.py` | FP32 vs INT8 quality comparison with IoU metrics |
| `scripts/quantize_int8.py` | Simple per-channel quantization (without calibration, for reference) |
| `scripts/quantize_int8_calibrated.py` | Manual calibration approach (superseded by PTQ) |
| `wasm/src/lib.rs` | WASM loader with INT8 dequantization |
| `weights/birefnet_int8.safetensors` | Quantized weights (not committed — generate with scripts) |
