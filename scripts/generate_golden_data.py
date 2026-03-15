"""Generate golden test data for candle-birefnet.

Runs the official BiRefNet model (inference mode) and saves:
- Input tensor
- All model weights (with candle-compatible key names)
- Expected output
"""

import sys
import os
import torch
import torch.nn as nn
from safetensors.torch import save_file

sys.path.insert(0, os.path.expanduser("~/Documents/Projects/BiRefNet"))

# Disable SDPA for reproducibility
import config as birefnet_config
_original_init = birefnet_config.Config.__init__
def _patched_init(self, *a, **kw):
    _original_init(self, *a, **kw)
    self.SDPA_enabled = False
birefnet_config.Config.__init__ = _patched_init

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "test-data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BIREFNET_WEIGHTS = (
    "/tmp/birefnet_cache/models--ZhengPeng7--BiRefNet/snapshots/"
    "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4/model.safetensors"
)


def generate_birefnet_inference():
    """Run full BiRefNet inference and save golden data."""
    from safetensors import safe_open
    from models.birefnet import BiRefNet

    model = BiRefNet(bb_pretrained=False)

    # Load pretrained weights
    f = safe_open(BIREFNET_WEIGHTS, framework="pt")
    state = model.state_dict()
    loaded = 0
    for key in f.keys():
        tensor = f.get_tensor(key).float()
        if key in state and tensor.shape == state[key].shape:
            state[key] = tensor
            loaded += 1
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {loaded}/{len(state)} weights")

    # Use a small input for tractable file size
    # BiRefNet default is 1024x1024 but we use 384x384 for testing
    torch.manual_seed(42)
    x = torch.randn(1, 3, 384, 384)

    with torch.no_grad():
        outputs = model(x)

    print(f"Input: {x.shape}")
    print(f"Output: {outputs[0].shape}")

    # Save with candle-compatible key names
    tensors = {"input": x, "output": outputs[0]}

    # Save model weights (skip relative_position_index buffers)
    for k, v in model.state_dict().items():
        if "relative_position_index" in k:
            continue
        tensors[f"param.{k}"] = v.float().contiguous()

    path = os.path.join(OUTPUT_DIR, "birefnet_full.safetensors")
    save_file(tensors, path)
    print(f"Saved {path} ({len(tensors)} tensors, {os.path.getsize(path)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    generate_birefnet_inference()
