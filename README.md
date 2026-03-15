# candle-birefnet

[BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (Bilateral Reference Network) inference for [Hugging Face Candle](https://github.com/huggingface/candle).

Pure Rust, no custom kernels — works on all Candle backends (CPU, CUDA, Metal, WASM).

## Results

PyTorch (left) vs **Candle/Rust** (right) using [`ZhengPeng7/BiRefNet`](https://huggingface.co/ZhengPeng7/BiRefNet) pretrained weights at **384x384** resolution.

Each panel shows: Input | Segmentation Mask | Composite

**Helicopter**

| PyTorch | Candle (Rust) |
|---------|---------------|
| ![PyTorch](examples/helicopter_result_pytorch.png) | ![Candle](examples/helicopter_result_candle.png) |

**Windmill**

| PyTorch | Candle (Rust) |
|---------|---------------|
| ![PyTorch](examples/windmill_result_pytorch.png) | ![Candle](examples/windmill_result_candle.png) |

*Sample images from [BiRefNet demo](https://huggingface.co/spaces/ZhengPeng7/BiRefNet_demo). Model: Swin-V1-Large backbone, default config.*

## Architecture

Default configuration: Swin-V1-Large backbone + ASPPDeformable decoder.

Depends on:
- [candle-swin](https://github.com/developer0hye/candle-swin) — Swin Transformer V1 backbone
- [candle-dcnv2](https://github.com/developer0hye/candle-dcnv2) — Deformable Convolution V2

## Quick Start

```bash
# Run inference on an image (downloads model automatically from HuggingFace)
cargo run --example inference --release -- --image your_image.jpg --size 384
```

### As a library

```rust
use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use candle_birefnet::BiRefNet;

let device = &Device::Cpu;
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&["model.safetensors"], DType::F32, device)?
};

let model = BiRefNet::new(vb)?;

// Input: [B, 3, H, W] ImageNet-normalized RGB tensor
let outputs = model.forward(&input)?;
// outputs[0]: [B, 1, H, W] segmentation logits (apply sigmoid for mask)
```

## Validation

End-to-end inference output matches PyTorch BiRefNet within **6.87e-5** max absolute error (fp32 precision limit).

## Reference

- [BiRefNet: Bilateral Reference for High-Resolution Dichotomous Image Segmentation](https://arxiv.org/abs/2401.03407)

## License

Apache-2.0
