"""Pix2Pix Facades inference demo.

This script performs inference only: it loads a trained Pix2Pix generator and
turns an architectural label map into a generated facade image.

Examples:
    python demo.py --input samples/label_1.png --output result.png
    python demo/demo.py --input demo/samples/label_1.png --output result.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

IMG_SIZE = 256
DEMO_DIR = Path(__file__).resolve().parent


def down(in_c, out_c, norm=True):
    layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up(in_c, out_c, dropout=False):
    layers = [
        nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
        nn.InstanceNorm2d(out_c),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    """U-Net generator with skip connections for 256x256 RGB inputs."""

    def __init__(self):
        super().__init__()
        self.d1 = down(3, 64, norm=False)
        self.d2 = down(64, 128)
        self.d3 = down(128, 256)
        self.d4 = down(256, 512)
        self.d5 = down(512, 512)
        self.d6 = down(512, 512)
        self.d7 = down(512, 512)
        self.d8 = down(512, 512, norm=False)

        self.u1 = up(512, 512, dropout=True)
        self.u2 = up(1024, 512, dropout=True)
        self.u3 = up(1024, 512, dropout=True)
        self.u4 = up(1024, 512)
        self.u5 = up(1024, 256)
        self.u6 = up(512, 128)
        self.u7 = up(256, 64)
        self.u8 = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7], 1))
        u3 = self.u3(torch.cat([u2, d6], 1))
        u4 = self.u4(torch.cat([u3, d5], 1))
        u5 = self.u5(torch.cat([u4, d4], 1))
        u6 = self.u6(torch.cat([u5, d3], 1))
        u7 = self.u7(torch.cat([u6, d2], 1))
        return self.u8(torch.cat([u7, d1], 1))


def load_generator(weights_path, device):
    model = UNetGenerator().to(device)
    state = torch.load(weights_path, map_location=device)
    state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
    model.load_state_dict(state)
    # Pix2Pix keeps dropout active at inference time as a stochastic noise source.
    model.train()
    return model


def load_label_map(input_path):
    img = Image.open(input_path).convert("RGB")
    w, h = img.size
    if w == 2 * h:
        img = img.crop((w // 2, 0, w, h))
    return img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)


@torch.no_grad()
def translate(input_path, model, device):
    label = load_label_map(input_path)
    arr = np.asarray(label, dtype=np.float32) / 127.5 - 1.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    out = model(x)[0]
    out = ((out + 1.0) / 2.0).clamp(0, 1)
    out = out.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((out * 255).astype(np.uint8))


def parse_args():
    parser = argparse.ArgumentParser(description="Pix2Pix Facades inference demo")
    parser.add_argument("--input", required=True, help="path to a label map image")
    parser.add_argument(
        "--output",
        default="result.png",
        help="path where the generated image is written",
    )
    parser.add_argument(
        "--weights",
        default=str(DEMO_DIR / "models" / "generator.pth"),
        help="path to the trained generator weights",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    weights_path = Path(args.weights)

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_generator(weights_path, device)
    result = translate(input_path, model, device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Saved generated image to: {output_path}")


if __name__ == "__main__":
    main()
