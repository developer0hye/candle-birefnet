# candle-birefnet

[BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (Bilateral Reference Network) inference for [Hugging Face Candle](https://github.com/huggingface/candle).

Pure Rust, no custom kernels — works on all Candle backends (CPU, CUDA, Metal, WASM).

## Architecture

Default configuration: Swin-V1-Large backbone + ASPPDeformable decoder.

Depends on:
- [candle-swin](https://github.com/developer0hye/candle-swin) — Swin Transformer V1 backbone
- [candle-dcnv2](https://github.com/developer0hye/candle-dcnv2) — Deformable Convolution V2

## Usage

```rust
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_birefnet::BiRefNet;

// Load pretrained weights from safetensors
let device = &Device::Cpu;
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&["model.safetensors"], DType::F32, device)?
};

let model = BiRefNet::new(vb)?;

// Input: [B, 3, H, W] normalized RGB image
let input = Tensor::randn(0f32, 1.0, (1, 3, 1024, 1024), device)?;
let outputs = model.forward(&input)?;
// outputs[0]: [B, 1, H, W] segmentation mask
```

## Validation

End-to-end inference output matches PyTorch BiRefNet within **6.87e-5** max absolute error (fp32 precision limit).

## Reference

- [BiRefNet: Bilateral Reference for High-Resolution Dichotomous Image Segmentation](https://arxiv.org/abs/2401.03407)

## License

Apache-2.0
