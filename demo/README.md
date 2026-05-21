# Pix2Pix Facades Inference Demo

This folder contains the inference-only demo for the Computer Vision II
image-to-image translation project.

Given an architectural label map, the trained Pix2Pix generator produces a
realistic facade image.

Team: Lascaro Gianluca, Wassim Bellani

## Contents

```text
demo.py
requirements.txt
models/generator.pth
samples/label_1.png
samples/label_2.png
samples/label_3.png
```

`models/generator.pth` is distributed as a GitHub Release asset because it is
larger than GitHub's normal 100 MB file limit. Download it from:

https://github.com/byluca/lascaro-bellani-pix2pix-facades/releases/tag/v1.0-model

Then place it in `models/generator.pth` before running inference.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

From this `demo/` folder:

```bash
python demo.py --input samples/label_1.png --output result.png
```

From the repository root:

```bash
python demo/demo.py --input demo/samples/label_1.png --output result.png
```

The input can be either a 256x256 label map or a full 512x256 Facades paired
sample. If a paired sample is provided, the script automatically extracts the
right half as the label map.

## Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--input` | Path to the input label map | required |
| `--output` | Path where the generated image is saved | `result.png` |
| `--weights` | Path to trained generator weights | `models/generator.pth` |

## Model

The model is a Pix2Pix conditional GAN trained on the Facades dataset:

- Generator: 8-level U-Net with skip connections and instance normalization.
- Discriminator during training: 70x70 PatchGAN.
- Objective: adversarial loss plus L1 reconstruction loss.
- Improvement: random jitter and horizontal flip data augmentation.

Full training, evaluation, metrics, and visual comparisons are available in
`Lascaro-Bellani.ipynb`.
