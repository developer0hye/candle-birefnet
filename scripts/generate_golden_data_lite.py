"""Generate golden test data for candle-birefnet (lite variant, Swin-V1-Tiny).

Runs the official BiRefNet_lite model (inference mode) and saves:
- Input tensor
- All model weights (with candle-compatible key names)
- Expected output

Usage:
    pip install torch safetensors transformers
    python scripts/generate_golden_data_lite.py
"""

import sys
import os
import torch

sys.path.insert(0, os.path.expanduser("~/Documents/Projects/BiRefNet"))

# Disable SDPA for reproducibility
import config as birefnet_config

_original_init = birefnet_config.Config.__init__


def _patched_init(self, *a, **kw):
    _original_init(self, *a, **kw)
    self.SDPA_enabled = False
    # Override backbone to swin_v1_t (tiny) for BiRefNet_lite
    self.bb = "swin_v1_t"
    self.lateral_channels_in_collection = [768, 384, 192, 96]
    if self.mul_scl_ipt == "cat":
        self.lateral_channels_in_collection = [
            ch * 2 for ch in self.lateral_channels_in_collection
        ]
    self.cxt = self.lateral_channels_in_collection[1:][::-1][-self.cxt_num :] if self.cxt_num else []


birefnet_config.Config.__init__ = _patched_init

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "test-data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_birefnet_lite_inference():
    """Run full BiRefNet_lite inference and save golden data."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file
    from models.birefnet import BiRefNet

    model = BiRefNet(bb_pretrained=False)

    # Download and load pretrained weights
    model_path = hf_hub_download(
        repo_id="ZhengPeng7/BiRefNet_lite", filename="model.safetensors"
    )
    print(f"Model path: {model_path}")

    f = safe_open(model_path, framework="pt")
    state = model.state_dict()
    loaded = 0
    for key in f.keys():
        tensor = f.get_tensor(key).float()
        if key in state and tensor.shape == state[key].shape:
            state[key] = tensor
            loaded += 1
        else:
            if key in state:
                print(f"  Shape mismatch: {key} model={state[key].shape} file={tensor.shape}")
            else:
                print(f"  Missing key: {key}")
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {loaded}/{len(state)} weights")

    # 384x384 input for tractable file size
    torch.manual_seed(42)
    x = torch.randn(1, 3, 384, 384)

    with torch.no_grad():
        outputs = model(x)

    print(f"Input: {x.shape}")
    print(f"Output: {outputs[0].shape}")

    # Save with candle-compatible key names
    tensors = {"input": x, "output": outputs[0]}

    for k, v in model.state_dict().items():
        if "relative_position_index" in k:
            continue
        tensors[f"param.{k}"] = v.float().contiguous()

    path = os.path.join(OUTPUT_DIR, "birefnet_lite.safetensors")
    save_file(tensors, path)
    print(
        f"Saved {path} ({len(tensors)} tensors, {os.path.getsize(path)/1024/1024:.1f} MB)"
    )


if __name__ == "__main__":
    generate_birefnet_lite_inference()
