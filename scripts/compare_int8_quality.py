#!/usr/bin/env python3
"""Compare BiRefNet_lite FP16 vs INT8 segmentation quality.

Loads both weight variants, runs inference on test images,
and produces side-by-side comparison + numerical metrics.

Usage:
    python scripts/compare_int8_quality.py
"""

import sys
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file

sys.path.insert(0, os.path.expanduser("~/Documents/Projects/BiRefNet"))

import config as birefnet_config

_original_init = birefnet_config.Config.__init__


def _patched_init(self, *a, **kw):
    _original_init(self, *a, **kw)
    self.SDPA_enabled = False
    self.bb = "swin_v1_t"
    self.lateral_channels_in_collection = [768, 384, 192, 96]
    if self.mul_scl_ipt == "cat":
        self.lateral_channels_in_collection = [
            ch * 2 for ch in self.lateral_channels_in_collection
        ]
    self.cxt = (
        self.lateral_channels_in_collection[1:][::-1][-self.cxt_num :]
        if self.cxt_num
        else []
    )


birefnet_config.Config.__init__ = _patched_init

from models.birefnet import BiRefNet


def dequantize_int8_weights(
    int8_tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Dequantize INT8 weights back to FP32 using stored scale/zero_point."""
    result: dict[str, torch.Tensor] = {}
    processed_keys: set[str] = set()

    for name, tensor in int8_tensors.items():
        if name.endswith(".__scale") or name.endswith(".__zero_point"):
            processed_keys.add(name)
            continue
        if name in processed_keys:
            continue

        scale_key = f"{name}.__scale"
        zp_key = f"{name}.__zero_point"

        if scale_key in int8_tensors:
            # Dequantize: x = (q - zero_point) * scale
            scale = int8_tensors[scale_key].float()
            zero_point = int8_tensors[zp_key].float()
            q = tensor.float()

            shape_for_broadcast = [-1] + [1] * (q.dim() - 1)
            scale_b = scale.reshape(shape_for_broadcast)
            zp_b = zero_point.reshape(shape_for_broadcast)

            result[name] = ((q - zp_b) * scale_b).float()
        else:
            result[name] = tensor.float()

    return result


def load_model_with_weights(weights: dict[str, torch.Tensor]) -> BiRefNet:
    """Load BiRefNet_lite with given weight dict."""
    model = BiRefNet(bb_pretrained=False)
    state = model.state_dict()
    loaded = 0
    for key in weights:
        if key in state and weights[key].shape == state[key].shape:
            state[key] = weights[key].float()
            loaded += 1
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded {loaded}/{len(state)} weights")
    return model


def run_inference(model: BiRefNet, image_path: str, size: int = 512) -> np.ndarray:
    """Run segmentation and return mask as numpy array [0, 1]."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)

    mask = torch.sigmoid(outputs[0]).squeeze().cpu().numpy()
    return mask


def create_comparison(
    input_path: str,
    mask_fp16: np.ndarray,
    mask_int8: np.ndarray,
    output_path: str,
    size: int = 512,
) -> None:
    """Create side-by-side comparison: Input | FP16 | INT8 | Diff."""
    img = Image.open(input_path).convert("RGB").resize((size, size))
    img_np = np.array(img)

    # Masks to RGB
    fp16_rgb = np.stack([mask_fp16 * 255] * 3, axis=-1).astype(np.uint8)
    int8_rgb = np.stack([mask_int8 * 255] * 3, axis=-1).astype(np.uint8)

    # Difference (amplified 10x for visibility)
    diff = np.abs(mask_fp16 - mask_int8)
    diff_amplified = np.clip(diff * 10, 0, 1)
    diff_rgb = np.stack([diff_amplified * 255] * 3, axis=-1).astype(np.uint8)

    # Composites
    fp16_composite = (img_np * mask_fp16[..., None] + 255 * (1 - mask_fp16[..., None])).astype(
        np.uint8
    )
    int8_composite = (img_np * mask_int8[..., None] + 255 * (1 - mask_int8[..., None])).astype(
        np.uint8
    )

    # Canvas: Input | FP16 Mask | INT8 Mask | Diff(10x) | FP16 Composite | INT8 Composite
    canvas = np.concatenate(
        [img_np, fp16_rgb, int8_rgb, diff_rgb, fp16_composite, int8_composite],
        axis=1,
    )
    Image.fromarray(canvas).save(output_path)
    print(f"  Saved: {output_path}")


def main() -> None:
    examples_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
    weights_dir = os.path.join(os.path.dirname(__file__), "..", "weights")
    int8_path = os.path.join(weights_dir, "birefnet_int8.safetensors")

    if not os.path.exists(int8_path):
        print(f"INT8 weights not found: {int8_path}")
        print("Run: python scripts/quantize_int8.py")
        return

    # Load FP16 model
    print("Loading FP16 model...")
    fp16_path = hf_hub_download(
        repo_id="ZhengPeng7/BiRefNet_lite", filename="model.safetensors"
    )
    fp16_weights = load_file(fp16_path)
    model_fp16 = load_model_with_weights(fp16_weights)

    # Load INT8 model (dequantize to FP32)
    print("Loading INT8 model (dequantized)...")
    int8_raw = load_file(int8_path)
    int8_weights = dequantize_int8_weights(int8_raw)
    model_int8 = load_model_with_weights(int8_weights)

    # Compare on test images
    test_images = ["Helicopter.jpg", "Windmill.jpg"]
    for image_name in test_images:
        image_path = os.path.join(examples_dir, image_name)
        if not os.path.exists(image_path):
            print(f"  Skipping {image_name} (not found)")
            continue

        stem = image_name.split(".")[0].lower()
        print(f"\nProcessing: {image_name}")

        mask_fp16 = run_inference(model_fp16, image_path)
        mask_int8 = run_inference(model_int8, image_path)

        # Metrics
        diff = np.abs(mask_fp16 - mask_int8)
        print(f"  Max diff:  {diff.max():.6f}")
        print(f"  Mean diff: {diff.mean():.6f}")
        print(f"  IoU:       {(np.sum((mask_fp16 > 0.5) & (mask_int8 > 0.5)) / np.sum((mask_fp16 > 0.5) | (mask_int8 > 0.5))):.4f}")

        output_path = os.path.join(
            examples_dir, f"{stem}_fp16_vs_int8_comparison.png"
        )
        create_comparison(image_path, mask_fp16, mask_int8, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
