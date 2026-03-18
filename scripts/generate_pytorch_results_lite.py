"""Generate PyTorch BiRefNet_lite inference result images for README comparison."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.expanduser("~/Documents/Projects/BiRefNet"))

import config as birefnet_config

_original_init = birefnet_config.Config.__init__

def _patched_init(self, *a, **kw):
    _original_init(self, *a, **kw)
    self.SDPA_enabled = False
    self.bb = "swin_v1_t"
    self.lateral_channels_in_collection = [768, 384, 192, 96]
    if self.mul_scl_ipt == "cat":
        self.lateral_channels_in_collection = [ch * 2 for ch in self.lateral_channels_in_collection]
    self.cxt = self.lateral_channels_in_collection[1:][::-1][-self.cxt_num :] if self.cxt_num else []

birefnet_config.Config.__init__ = _patched_init

import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from models.birefnet import BiRefNet


def run_inference(model, image_path, size):
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
    return img.resize((size, size)), mask


def create_comparison(input_img, mask, output_path):
    w, h = input_img.size
    input_np = np.array(input_img)
    mask_rgb = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)

    alpha = mask[..., None]
    composite = (input_np * alpha + 255 * (1 - alpha)).astype(np.uint8)

    canvas = np.concatenate([input_np, mask_rgb, composite], axis=1)
    Image.fromarray(canvas).save(output_path)
    print(f"Saved: {output_path}")


def main():
    model_path = hf_hub_download(repo_id="ZhengPeng7/BiRefNet_lite", filename="model.safetensors")
    model = BiRefNet(bb_pretrained=False)
    f = safe_open(model_path, framework="pt")
    state = model.state_dict()
    for key in f.keys():
        tensor = f.get_tensor(key).float()
        if key in state and tensor.shape == state[key].shape:
            state[key] = tensor
    model.load_state_dict(state)
    model.eval()

    examples_dir = os.path.join(os.path.dirname(__file__), "..", "examples")

    for image_name in ["Helicopter", "Windmill"]:
        image_path = os.path.join(examples_dir, f"{image_name}.jpg")
        stem = image_name.lower()

        for size in [1024, 384]:
            img, mask = run_inference(model, image_path, size)
            output_path = os.path.join(examples_dir, f"{stem}_result_pytorch_lite_{size}.png")
            create_comparison(img, mask, output_path)


if __name__ == "__main__":
    main()
