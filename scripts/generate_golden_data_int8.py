"""Generate golden test data for INT8 dequantize validation.

Loads INT8 weights, dequantizes in Python, runs inference,
saves input + output + INT8 weights for Rust-side comparison.

Usage:
    python scripts/generate_golden_data_int8.py
"""

import sys
import os

import torch

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

from safetensors.torch import load_file, save_file
from models.birefnet import BiRefNet


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "test-data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def dequantize_int8_weights(int8_tensors):
    """Dequantize INT8 weights back to FP32 — same logic as Rust side."""
    result = {}
    for name, tensor in int8_tensors.items():
        if name.endswith(".__scale") or name.endswith(".__zero_point"):
            continue
        scale_key = f"{name}.__scale"
        zp_key = f"{name}.__zero_point"
        if scale_key in int8_tensors:
            scale = int8_tensors[scale_key].float()
            zp = int8_tensors[zp_key].float()
            q = tensor.float()
            shape = [-1] + [1] * (q.dim() - 1)
            result[name] = ((q - zp.reshape(shape)) * scale.reshape(shape)).float()
        else:
            result[name] = tensor.float()
    return result


def main():
    weights_path = os.path.join(
        os.path.dirname(__file__), "..", "weights", "birefnet_int8.safetensors"
    )
    if not os.path.exists(weights_path):
        print(f"INT8 weights not found: {weights_path}")
        print("Run: python scripts/quantize_int8_ptq.py")
        return

    # Load and dequantize INT8 weights
    print("Loading INT8 weights and dequantizing...")
    int8_raw = load_file(weights_path)
    weights = dequantize_int8_weights(int8_raw)

    # Build model with dequantized weights
    model = BiRefNet(bb_pretrained=False)
    state = model.state_dict()
    loaded = 0
    for key in weights:
        if key in state and weights[key].shape == state[key].shape:
            state[key] = weights[key]
            loaded += 1
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {loaded}/{len(state)} weights")

    # Generate input
    torch.manual_seed(42)
    x = torch.randn(1, 3, 384, 384)

    with torch.no_grad():
        outputs = model(x)

    print(f"Input: {x.shape}")
    print(f"Output: {outputs[0].shape}")

    # Save: input, output, and all INT8 tensors (weights + scale + zp)
    # The Rust test will load the same INT8 file, dequantize, and compare output
    tensors = {"input": x, "output": outputs[0]}

    # Save dequantized weights (for Rust VarMap loading)
    for k, v in state.items():
        if "relative_position_index" in k:
            continue
        tensors[f"param.{k}"] = v.float().contiguous()

    path = os.path.join(OUTPUT_DIR, "birefnet_int8_dequantized.safetensors")
    save_file(tensors, path)
    print(
        f"Saved {path} ({len(tensors)} tensors, {os.path.getsize(path)/1024/1024:.1f} MB)"
    )


if __name__ == "__main__":
    main()
