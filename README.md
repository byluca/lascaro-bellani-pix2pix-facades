# Lascaro-Bellani Pix2Pix Facades

Computer Vision II project: image-to-image translation on the Facades dataset
using Pix2Pix.

Team: Lascaro Gianluca, Wassim Bellani

## Files

```text
Lascaro-Bellani.ipynb          training, evaluation, and demo packaging notebook
Lascaro-Bellani-report.pdf     project report
requirements.txt               notebook dependencies
demo/                           inference-only demo
outputs/                        figures generated from the final run
```

## Final Results

```text
Metric   Baseline   Improved
PSNR      12.3209    13.0348
SSIM       0.2049     0.2543
L1         0.1901     0.1723
LPIPS      0.5543     0.5146
FID      180.4805   135.5984
```

The improved model uses data augmentation. The trained model is distributed as
the `generator.pth` asset in the repository release because it is larger than
GitHub's normal 100 MB file limit.

## Demo

From the repository root:

```bash
pip install -r demo/requirements.txt
python demo/demo.py --input demo/samples/label_1.png --output result.png
```

Before running the demo, download `generator.pth` from the repository Release:

https://github.com/byluca/lascaro-bellani-pix2pix-facades/releases/tag/v1.0-model

Place it at:

```text
demo/models/generator.pth
```
