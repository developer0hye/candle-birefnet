"""Generate INT8 dequantized inference result images for README."""

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
from safetensors.torch import load_file
from models.birefnet import BiRefNet


def dequantize_int8_weights(int8_tensors):
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
    examples_dir = os.path.join(os.path.dirname(__file__), "..", "examples")
    weights_dir = os.path.join(os.path.dirname(__file__), "..", "weights")
    int8_path = os.path.join(weights_dir, "birefnet_int8.safetensors")

    # Load INT8 model
    print("Loading INT8 model...")
    int8_raw = load_file(int8_path)
    weights = dequantize_int8_weights(int8_raw)
    model = BiRefNet(bb_pretrained=False)
    state = model.state_dict()
    for key in weights:
        if key in state and weights[key].shape == state[key].shape:
            state[key] = weights[key]
    model.load_state_dict(state)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for image_name in ["Helicopter", "Windmill"]:
        image_path = os.path.join(examples_dir, f"{image_name}.jpg")
        stem = image_name.lower()

        img = Image.open(image_path).convert("RGB").resize((512, 512))
        input_tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
        mask = torch.sigmoid(output[0]).squeeze().cpu().numpy()

        img_np = np.array(img)
        mask_rgb = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)
        composite = (img_np * mask[..., None] + 255 * (1 - mask[..., None])).astype(np.uint8)
        canvas = np.concatenate([img_np, mask_rgb, composite], axis=1)

        out_path = os.path.join(examples_dir, f"{stem}_result_int8_512.png")
        Image.fromarray(canvas).save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
