## Report figures

This folder contains tooling to generate plots for the IEEE-style report.

### Install Python deps

```bash
pip install numpy pillow matplotlib
```

### Generate plots (PSNR / ratio / FPS vs quality)

Assuming you have input frames in `Frames/` and a built executable in `build/`:

```bash
python3 report/plots_flipbook.py --input-dir Frames --build-dir build --qualities 10,20,30,40,50,60,70,80,90
```

Output images are saved to `report/figures/`:

- `psnr_vs_quality.png`
- `ratio_vs_quality.png`
- `psnr_vs_ratio.png`
- `fps_vs_quality.png`

