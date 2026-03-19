#!/usr/bin/env python3
"""Post-Training Static Quantization for BiRefNet_lite using torch.ao.quantization.

Uses PyTorch's built-in quantization framework:
1. Fuse Conv+BN+ReLU modules
2. Insert quantization observers
3. Calibrate with DUTS-TE dataset
4. Convert to quantized model
5. Extract INT8 weights + scale/zero_point
6. Save as SafeTensors for candle dequantize loader

Usage:
    python scripts/quantize_int8_ptq.py \
        --dataset-dir /tmp/duts-te/DUTS-TE/DUTS-TE-Image \
        --num-calibration 500
"""

import argparse
import os
import sys
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.ao.quantization as quant
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

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

# Set quantization backend (qnnpack for ARM/macOS, fbgemm for x86)
torch.backends.quantized.engine = "qnnpack"


def load_calibration_images(
    dataset_dir: Optional[str], num_images: int, size: int = 512
):
    """Load calibration images from dataset directory."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data = []

    if dataset_dir and os.path.isdir(dataset_dir):
        random.seed(42)
        image_files = sorted(
            f for f in os.listdir(dataset_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
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
        print("  No dataset dir, using random calibration data")
        torch.manual_seed(42)
        norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        for _ in range(num_images):
            data.append(norm(torch.rand(3, size, size)).unsqueeze(0))

    return data[:num_images]


def extract_quantized_weights(model_fp32, model_quantized):
    """Extract INT8 weights + scale/zero_point from quantized model.

    Walks the quantized model, finds quantized layers, and extracts:
    - INT8 weight tensor
    - Per-channel scale (FP32)
    - Per-channel zero_point (INT8)

    Non-quantized parameters are stored as FP16.
    """
    output = {}
    fp32_state = model_fp32.state_dict()

    # Walk quantized model to find quantized weights
    quantized_keys = set()
    for name, module in model_quantized.named_modules():
        weight_key = f"{name}.weight"

        # Try to get quantized weight
        w = getattr(module, "weight", None)
        if w is None:
            continue

        # Check if it's a callable (quantized modules use weight() method)
        if callable(w):
            try:
                q_weight = w()
            except Exception:
                continue
        elif hasattr(w, "is_quantized") and w.is_quantized:
            q_weight = w
        else:
            continue

        if not hasattr(q_weight, "is_quantized") or not q_weight.is_quantized:
            continue

        int_repr = q_weight.int_repr()  # INT8 tensor
        q_scheme = q_weight.qscheme()

        if q_scheme in (torch.per_channel_affine, torch.per_channel_symmetric):
            scales = q_weight.q_per_channel_scales().float()
            zero_points = q_weight.q_per_channel_zero_points().to(torch.int8)
        elif q_scheme == torch.per_tensor_affine:
            scales = torch.tensor([q_weight.q_scale()], dtype=torch.float32)
            zero_points = torch.tensor([q_weight.q_zero_point()], dtype=torch.int8)
        else:
            output[weight_key] = fp32_state[weight_key].half()
            continue

        output[weight_key] = int_repr.contiguous()
        output[f"{weight_key}.__scale"] = scales.contiguous()
        output[f"{weight_key}.__zero_point"] = zero_points.contiguous()
        quantized_keys.add(weight_key)

    # Add all non-quantized parameters as FP16
    for key, tensor in fp32_state.items():
        if key not in quantized_keys and key not in output:
            output[key] = tensor.half().contiguous()

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--num-calibration", type=int, default=500)
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
    model = BiRefNet(bb_pretrained=False)
    f = safe_open(model_path, framework="pt")
    state = model.state_dict()
    for key in f.keys():
        tensor = f.get_tensor(key).float()
        if key in state and tensor.shape == state[key].shape:
            state[key] = tensor
    model.load_state_dict(state)
    model.eval()

    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Original model: {original_size / 1024 / 1024:.1f} MB")

    # Keep a copy of fp32 state for extraction
    model_fp32_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Step 1: Set quantization config
    # Per-channel weight quantization + per-tensor activation quantization
    print("Configuring quantization...")
    model.qconfig = quant.QConfig(
        activation=quant.HistogramObserver.with_args(
            dtype=torch.quint8, reduce_range=False
        ),
        weight=quant.PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        ),
    )

    # Step 2: Prepare model (insert observers)
    print("Preparing model with observers...")
    model_prepared = quant.prepare(model, inplace=False)

    # Step 3: Calibrate
    print(f"Calibrating with {args.num_calibration} images...")
    calibration_data = load_calibration_images(
        args.dataset_dir, args.num_calibration
    )

    with torch.no_grad():
        for i, x in enumerate(calibration_data):
            try:
                model_prepared(x)
            except Exception as e:
                print(f"  Warning: calibration image {i} failed: {e}")
            if (i + 1) % 50 == 0:
                print(f"  Calibration: {i + 1}/{len(calibration_data)}")

    # Step 4: Convert to quantized model
    print("Converting to quantized model...")
    model_quantized = quant.convert(model_prepared, inplace=False)

    # Step 5: Extract weights
    print("Extracting quantized weights...")

    # Rebuild fp32 model for state dict reference
    model_fp32 = BiRefNet(bb_pretrained=False)
    model_fp32.load_state_dict(model_fp32_state)

    output_tensors = extract_quantized_weights(model_fp32, model_quantized)

    n_int8 = sum(1 for k in output_tensors if output_tensors[k].dtype == torch.int8 and not k.endswith(".__zero_point"))
    n_fp16 = sum(1 for k, v in output_tensors.items() if v.dtype == torch.float16)
    quantized_size = sum(t.numel() * t.element_size() for t in output_tensors.values())

    print(f"\nResults:")
    print(f"  INT8 weight tensors: {n_int8}")
    print(f"  FP16 tensors: {n_fp16}")
    print(f"  Scale/ZP tensors: {sum(1 for k in output_tensors if '__scale' in k or '__zero_point' in k)}")
    print(f"  Size: {original_size / 1024 / 1024:.1f} MB → {quantized_size / 1024 / 1024:.1f} MB")

    # Quality check: use compare_int8_quality.py separately (dequantize → FP32 inference)
    # PyTorch eager quantization can't run BiRefNet's complex forward pass directly
    print("\nSkipping in-process quality check (use compare_int8_quality.py instead)")

    # Save
    print(f"\nSaving: {output_path}")
    save_file(output_tensors, output_path)
    actual_size = os.path.getsize(output_path)
    print(f"File size: {actual_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
