# candle-birefnet

[BiRefNet](https://github.com/ZhengPeng7/BiRefNet) (Bilateral Reference Network) inference for [Hugging Face Candle](https://github.com/huggingface/candle).

Pure Rust, no custom kernels — works on all Candle backends (CPU, CUDA, Metal, WASM).

## Results

PyTorch (left) vs **Candle/Rust** (right) using [`ZhengPeng7/BiRefNet`](https://huggingface.co/ZhengPeng7/BiRefNet) pretrained weights. Each panel shows: Input | Segmentation Mask | Composite.

Model: Swin-V1-Large backbone, default config.

### 1024x1024

| PyTorch | Candle (Rust) |
|---------|---------------|
| ![PyTorch](examples/helicopter_result_pytorch_1024.png) | ![Candle](examples/helicopter_result_candle_1024.png) |
| ![PyTorch](examples/windmill_result_pytorch_1024.png) | ![Candle](examples/windmill_result_candle_1024.png) |

### 384x384

| PyTorch | Candle (Rust) |
|---------|---------------|
| ![PyTorch](examples/helicopter_result_pytorch_384.png) | ![Candle](examples/helicopter_result_candle_384.png) |
| ![PyTorch](examples/windmill_result_pytorch_384.png) | ![Candle](examples/windmill_result_candle_384.png) |

*Sample images from [BiRefNet demo](https://huggingface.co/spaces/ZhengPeng7/BiRefNet_demo).*

## Architecture

Default configuration: Swin-V1-Large backbone + ASPPDeformable decoder.

Depends on:
- [candle-swin](https://github.com/developer0hye/candle-swin) — Swin Transformer V1 backbone
- [candle-dcnv2](https://github.com/developer0hye/candle-dcnv2) — Deformable Convolution V2

## Quick Start

```bash
# Run inference on an image (downloads model automatically from HuggingFace)
cargo run --example inference --release -- --image your_image.jpg --size 1024
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

End-to-end inference output matches PyTorch BiRefNet:
- **384x384**: max error 6.87e-5
- **1024x1024**: max error 1.63e-4

## Note on candle-core Conv2d Bug

This project uses a [patched candle-core](https://github.com/developer0hye/candle/tree/fix/conv2d-tiled-bug) that works around a `conv2d_tiled` bug ([huggingface/candle#3404](https://github.com/huggingface/candle/issues/3404)). The patch switches the default Conv2d implementation from `TiledIm2Col` to `FullIm2Col`. Once the upstream fix is merged, this project will switch back to the official candle release.

## Reference

- [BiRefNet: Bilateral Reference for High-Resolution Dichotomous Image Segmentation](https://arxiv.org/abs/2401.03407)

## License

Apache-2.0
