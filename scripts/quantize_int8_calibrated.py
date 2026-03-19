#!/usr/bin/env python3
"""Calibration-based INT8 quantization for BiRefNet_lite.

Uses PyTorch's post-training static quantization with calibration data
to find optimal scale/zero_point per layer.

Flow:
1. Load FP32 model
2. Insert quantization observers
3. Run calibration (forward pass on representative images)
4. Convert to quantized model
5. Extract INT8 weights + scale/zero_point
6. Save as SafeTensors (compatible with candle dequantize loader)

Usage:
    python scripts/quantize_int8_calibrated.py [--num-calibration 50] [--output PATH]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torchvision import transforms

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


def generate_calibration_data(
    num_images: int, size: int = 512, dataset_dir: str = None
) -> list[torch.Tensor]:
    """Load calibration images from a dataset directory (e.g., DUTS-TE).

    Falls back to synthetic data if no dataset is provided.
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data: list[torch.Tensor] = []

    if dataset_dir and os.path.isdir(dataset_dir):
        import random

        random.seed(42)
        image_files = sorted(
            f for f in os.listdir(dataset_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))
        )
        sampled = random.sample(image_files, min(num_images, len(image_files)))
        print(f"  Sampling {len(sampled)} images from {dataset_dir}")

        for fname in sampled:
            try:
                img = Image.open(os.path.join(dataset_dir, fname)).convert("RGB")
                data.append(transform(img).unsqueeze(0))
            except Exception as e:
                print(f"  Warning: failed to load {fname}: {e}")
    else:
        # Fallback: synthetic + example images
        print("  No dataset dir provided, using synthetic calibration data")
        examples_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
        for fname in ["Helicopter.jpg", "Windmill.jpg"]:
            path = os.path.join(examples_dir, fname)
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                data.append(transform(img).unsqueeze(0))

        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        torch.manual_seed(42)
        while len(data) < num_images:
            t = torch.rand(3, size, size)
            data.append(norm(t).unsqueeze(0))

    return data[:num_images]


def quantize_weights_with_calibration(
    model: BiRefNet,
    calibration_data: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Run calibration and extract per-channel quantized weights.

    Instead of using PyTorch's full quantization framework (which changes
    the model graph and is hard to export), we:
    1. Run calibration to collect activation statistics
    2. Use the statistics to determine which layers are sensitive
    3. Apply per-channel weight quantization with MinMax observers
    """
    model.eval()

    # Collect per-layer weight statistics during calibration
    print(f"  Running calibration on {len(calibration_data)} images...")
    activation_ranges: dict[str, tuple[float, float]] = {}

    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                with torch.no_grad():
                    if name not in activation_ranges:
                        activation_ranges[name] = (
                            output.min().item(),
                            output.max().item(),
                        )
                    else:
                        old_min, old_max = activation_ranges[name]
                        activation_ranges[name] = (
                            min(old_min, output.min().item()),
                            max(old_max, output.max().item()),
                        )
        return hook_fn

    # Register hooks on all modules with weights
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Run calibration
    with torch.no_grad():
        for i, x in enumerate(calibration_data):
            try:
                model(x)
            except Exception as e:
                print(f"  Warning: calibration image {i} failed: {e}")
            if (i + 1) % 10 == 0:
                print(f"  Calibration: {i + 1}/{len(calibration_data)}")

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"  Collected activation ranges for {len(activation_ranges)} layers")

    # Identify sensitive layers (wide activation range = more sensitive to quantization)
    sensitive_layers: set[str] = set()
    for name, (act_min, act_max) in activation_ranges.items():
        act_range = act_max - act_min
        # Layers with very wide activation ranges are sensitive
        if act_range > 100:
            sensitive_layers.add(name)

    print(f"  Sensitive layers (kept FP16): {len(sensitive_layers)}")

    # Quantize weights
    state = model.state_dict()
    output_tensors: dict[str, torch.Tensor] = {}
    n_quantized = 0
    n_kept = 0

    for name, tensor in state.items():
        # Find parent module name
        parent_name = ".".join(name.split(".")[:-1])

        # Skip 1D tensors, small tensors, sensitive layers, position bias
        should_skip = (
            tensor.dim() < 2
            or tensor.numel() < 64
            or "relative_position_bias_table" in name
            or parent_name in sensitive_layers
        )

        if should_skip:
            output_tensors[name] = tensor.half()
            n_kept += 1
        else:
            w_q, scale, zero_point = quantize_per_channel_symmetric(tensor)
            output_tensors[name] = w_q
            output_tensors[f"{name}.__scale"] = scale
            output_tensors[f"{name}.__zero_point"] = zero_point
            n_quantized += 1

    return output_tensors, n_quantized, n_kept


def quantize_per_channel_symmetric(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric per-channel quantization (better for weights).

    Uses symmetric range [-max_abs, max_abs] → [-127, 127]
    Zero point is always 0, which simplifies dequantization.
    """
    w = weight.float()
    out_channels = w.shape[0]
    w_flat = w.reshape(out_channels, -1)

    # Per-channel max absolute value
    w_abs_max = w_flat.abs().max(dim=1).values  # [out_channels]
    w_abs_max = w_abs_max.clamp(min=1e-10)

    # Symmetric scale: map [-max, max] → [-127, 127]
    scale = w_abs_max / 127.0

    # Quantize
    shape_for_broadcast = [-1] + [1] * (w.dim() - 1)
    scale_b = scale.reshape(shape_for_broadcast)
    w_q = torch.round(w / scale_b).clamp(-127, 127).to(torch.int8)

    # Zero point is always 0 for symmetric quantization
    zero_point = torch.zeros(out_channels, dtype=torch.int8)

    return w_q, scale, zero_point


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-calibration", type=int, default=500)
    parser.add_argument("--dataset-dir", type=str, default=None,
                       help="Directory with calibration images (e.g., DUTS-TE/DUTS-TE-Image)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "..", "weights", "birefnet_int8.safetensors"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load model
    print("Loading BiRefNet_lite model...")
    model_path = hf_hub_download(
        repo_id="ZhengPeng7/BiRefNet_lite", filename="model.safetensors"
    )
    f = safe_open(model_path, framework="pt")
    model = BiRefNet(bb_pretrained=False)
    state = model.state_dict()
    for key in f.keys():
        tensor = f.get_tensor(key).float()
        if key in state and tensor.shape == state[key].shape:
            state[key] = tensor
    model.load_state_dict(state)
    model.eval()

    original_size = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    print(f"Original model: {original_size / 1024 / 1024:.1f} MB")

    # Generate calibration data
    print("Loading calibration data...")
    calibration_data = generate_calibration_data(
        args.num_calibration, dataset_dir=args.dataset_dir
    )

    # Quantize with calibration
    print("Quantizing with calibration...")
    output_tensors, n_quantized, n_kept = quantize_weights_with_calibration(
        model, calibration_data
    )

    quantized_size = sum(
        t.numel() * t.element_size() for t in output_tensors.values()
    )
    print(f"\nResults:")
    print(f"  INT8 tensors: {n_quantized}")
    print(f"  FP16 tensors: {n_kept}")
    print(f"  Size: {original_size / 1024 / 1024:.1f} MB → {quantized_size / 1024 / 1024:.1f} MB")

    print(f"Saving: {output_path}")
    save_file(output_tensors, output_path)
    actual_size = os.path.getsize(output_path)
    print(f"File size: {actual_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
