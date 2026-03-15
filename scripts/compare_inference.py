"""Compare BiRefNet inference: PyTorch vs Candle.

Runs both implementations on example images and saves:
- PyTorch mask output
- Input/weights/output as safetensors for Candle golden test
- Side-by-side comparison images for README
"""

import sys
import os
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file

sys.path.insert(0, os.path.expanduser("~/Documents/Projects/BiRefNet"))

import config as birefnet_config
_original_init = birefnet_config.Config.__init__
def _patched_init(self, *a, **kw):
    _original_init(self, *a, **kw)
    self.SDPA_enabled = False
birefnet_config.Config.__init__ = _patched_init

BIREFNET_WEIGHTS = (
    "/tmp/birefnet_cache/models--ZhengPeng7--BiRefNet/snapshots/"
    "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4/model.safetensors"
)
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")


def load_model():
    from safetensors import safe_open
    from models.birefnet import BiRefNet

    model = BiRefNet(bb_pretrained=False)
    f = safe_open(BIREFNET_WEIGHTS, framework="pt")
    state = model.state_dict()
    for key in f.keys():
        tensor = f.get_tensor(key).float()
        if key in state and tensor.shape == state[key].shape:
            state[key] = tensor
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess(image_path, size=(1024, 1024)):
    """Load and preprocess image to [1, 3, H, W] normalized tensor."""
    img = Image.open(image_path).convert("RGB").resize(size, Image.BILINEAR)
    # Convert PIL to tensor without numpy
    pixels = list(img.getdata())
    h, w = size[1], size[0]
    r = torch.tensor([p[0] for p in pixels], dtype=torch.float32).reshape(h, w) / 255.0
    g = torch.tensor([p[1] for p in pixels], dtype=torch.float32).reshape(h, w) / 255.0
    b = torch.tensor([p[2] for p in pixels], dtype=torch.float32).reshape(h, w) / 255.0
    tensor = torch.stack([r, g, b], dim=0)
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0), img


def postprocess_mask(mask_tensor, original_size=None):
    """Convert mask tensor to PIL Image."""
    mask = mask_tensor.sigmoid().squeeze().cpu()
    mask = (mask * 255).clamp(0, 255).to(torch.uint8)
    h, w = mask.shape
    mask_bytes = bytes(mask.flatten().tolist())
    mask_img = Image.frombytes("L", (w, h), mask_bytes)
    if original_size:
        mask_img = mask_img.resize(original_size, Image.BILINEAR)
    return mask_img


def create_comparison(input_img, mask_img, name):
    """Create side-by-side comparison: input | mask | composite."""
    w, h = input_img.size
    # Create composite (foreground on white background)
    mask_rgba = mask_img.convert("L")
    composite = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    fg = input_img.convert("RGBA")
    composite = Image.composite(fg, composite, mask_rgba)

    # Side by side: input | mask | composite
    total_w = w * 3
    canvas = Image.new("RGB", (total_w, h), (255, 255, 255))
    canvas.paste(input_img, (0, 0))
    canvas.paste(mask_img.convert("RGB"), (w, 0))
    canvas.paste(composite.convert("RGB"), (w * 2, 0))

    out_path = os.path.join(OUTPUT_DIR, f"{name}_result.png")
    canvas.save(out_path)
    print(f"  Saved comparison: {out_path}")
    return out_path


def save_candle_test_data(input_tensor, output_tensor, name):
    """Save input and expected output for Candle golden test."""
    save_file(
        {"input": input_tensor, "output": output_tensor},
        os.path.join(os.path.dirname(__file__), "..", "test-data", f"real_image_{name}.safetensors"),
    )


def main():
    model = load_model()

    for img_file in ["Helicopter.jpg", "Windmill.jpg"]:
        name = img_file.split(".")[0].lower()
        img_path = os.path.join(EXAMPLES_DIR, img_file)
        if not os.path.exists(img_path):
            print(f"Skipping {img_file} (not found)")
            continue

        print(f"\nProcessing {img_file}...")
        input_tensor, input_img = preprocess(img_path, size=(1024, 1024))

        with torch.no_grad():
            output = model(input_tensor)
        mask_tensor = output[-1]

        # Save mask
        mask_img = postprocess_mask(mask_tensor, original_size=input_img.size)
        mask_path = os.path.join(OUTPUT_DIR, f"{name}_mask.png")
        mask_img.save(mask_path)
        print(f"  Mask saved: {mask_path}")

        # Create comparison
        create_comparison(input_img, mask_img, name)

        # Save test data for Candle validation
        save_candle_test_data(input_tensor, mask_tensor, name)
        print(f"  Test data saved for Candle validation")

    print("\nDone!")


if __name__ == "__main__":
    main()
