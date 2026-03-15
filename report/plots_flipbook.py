#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import time
import re


def parse_args():
    p = argparse.ArgumentParser(description="Generate plots for flipbook codec report")
    p.add_argument("--build-dir", default="build", help="CMake build directory (default: build)")
    p.add_argument("--input-dir", default="Frames", help="Directory with input frames (default: Frames)")
    p.add_argument("--qualities", default="10,20,30,40,50,60,70,80,90",
                   help="Comma-separated JPEG-like quality values 1..100")
    p.add_argument("--runs", type=int, default=1, help="Runs per quality (default: 1)")
    p.add_argument("--keep-temp", action="store_true", help="Do not delete temp outputs")
    p.add_argument("--skip-build", action="store_true", help="Do not call cmake --build")
    return p.parse_args()


def ensure_deps():
    try:
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception:
        print("Missing Python deps. Install:")
        print("  pip install numpy pillow matplotlib")
        raise


def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def build_project(build_dir):
    run(["cmake", "--build", build_dir])


def list_frames(input_dir):
    files = []
    for name in os.listdir(input_dir):
        p = os.path.join(input_dir, name)
        if os.path.isfile(p):
            files.append(p)
    files.sort()
    if not files:
        raise RuntimeError(f"No frames found in {input_dir}")
    return files


def total_input_bytes(files):
    return sum(os.path.getsize(p) for p in files)


def compute_psnr_dir(orig_dir, recon_dir):
    import numpy as np
    from PIL import Image

    orig_files = [f for f in os.listdir(orig_dir) if os.path.isfile(os.path.join(orig_dir, f))]
    orig_files.sort()

    recon_files = [f for f in os.listdir(recon_dir) if os.path.isfile(os.path.join(recon_dir, f))]
    recon_files.sort()

    if not orig_files or not recon_files:
        raise RuntimeError("Empty orig/recon directory")

    # recon filenames are frame_XXXX.png; match by index if possible
    # fallback: compare by sorted order
    pairs = []
    by_index = {}
    rx = re.compile(r"frame_(\d+)\.png$")
    for f in recon_files:
        m = rx.search(f)
        if m:
            by_index[int(m.group(1))] = f

    if by_index:
        for i, orig_name in enumerate(orig_files):
            if i in by_index:
                pairs.append((orig_name, by_index[i]))
    if not pairs:
        pairs = list(zip(orig_files, recon_files))

    total_mse = 0.0
    total_pixels = 0
    for o, r in pairs:
        o_path = os.path.join(orig_dir, o)
        r_path = os.path.join(recon_dir, r)
        a = np.asarray(Image.open(o_path).convert("RGB"), dtype=np.float32)
        b = np.asarray(Image.open(r_path).convert("RGB"), dtype=np.float32)
        if a.shape != b.shape:
            raise RuntimeError(f"Shape mismatch: {o} {a.shape} vs {r} {b.shape}")
        d = a - b
        mse = float(np.mean(d * d))
        total_mse += mse * (a.shape[0] * a.shape[1] * a.shape[2])
        total_pixels += a.shape[0] * a.shape[1] * a.shape[2]

    mse = total_mse / max(1, total_pixels)
    if mse < 1e-12:
        return float("inf")
    psnr = 10.0 * np.log10((255.0 * 255.0) / mse)
    return float(psnr)


def main():
    args = parse_args()
    ensure_deps()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(project_root, args.build_dir)
    exe = os.path.join(build_dir, "askProjectCUDA")
    if os.name == "nt":
        exe += ".exe"

    input_dir = os.path.join(project_root, args.input_dir)
    frames = list_frames(input_dir)
    input_bytes = total_input_bytes(frames)
    frame_count = len(frames)

    print(f"Frames: {frame_count}, input bytes: {input_bytes}")
    print(f"Executable: {exe}")

    if not args.skip_build:
        try:
            build_project(build_dir)
        except Exception as e:
            print("[WARN] Build step failed, continuing with existing executable.")
            print("       ", str(e).splitlines()[0])

    qualities = [int(x.strip()) for x in args.qualities.split(",") if x.strip()]
    qualities = [q for q in qualities if 1 <= q <= 100]
    if not qualities:
        raise RuntimeError("No valid qualities provided")

    out_dir = os.path.join(project_root, "report", "figures")
    os.makedirs(out_dir, exist_ok=True)

    tmp_bin = os.path.join(project_root, "report", "_tmp_flipbook.bin")
    tmp_out = os.path.join(project_root, "report", "_tmp_out_frames")

    rows = []
    for q in qualities:
        psnrs = []
        ratios = []
        fpss = []

        for _ in range(args.runs):
            if os.path.exists(tmp_bin):
                os.remove(tmp_bin)
            if os.path.exists(tmp_out):
                shutil.rmtree(tmp_out)

            t0 = time.perf_counter()
            run([exe, "compress", "-q", str(q), input_dir, tmp_bin], cwd=project_root)
            t1 = time.perf_counter()
            run([exe, "decompress", tmp_bin, tmp_out], cwd=project_root)
            t2 = time.perf_counter()

            bin_size = os.path.getsize(tmp_bin)
            ratio = input_bytes / max(1, bin_size)
            fps = frame_count / max(1e-9, (t2 - t1))
            psnr = compute_psnr_dir(input_dir, tmp_out)

            psnrs.append(psnr)
            ratios.append(ratio)
            fpss.append(fps)

        row = {
            "quality": q,
            "psnr": sum(psnrs) / len(psnrs),
            "ratio": sum(ratios) / len(ratios),
            "fps": sum(fpss) / len(fpss),
        }
        rows.append(row)
        print(f"q={q:3d}  PSNR={row['psnr']:.2f} dB  ratio={row['ratio']:.2f}x  FPS(decomp)={row['fps']:.1f}")

    if not args.keep_temp:
        if os.path.exists(tmp_bin):
            os.remove(tmp_bin)
        if os.path.exists(tmp_out):
            shutil.rmtree(tmp_out)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qs = [r["quality"] for r in rows]
    ps = [r["psnr"] for r in rows]
    rs = [r["ratio"] for r in rows]
    fs = [r["fps"] for r in rows]

    plt.figure(figsize=(7, 4))
    plt.plot(qs, ps, marker="o")
    plt.xlabel("Quality (1..100)")
    plt.ylabel("PSNR [dB]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "psnr_vs_quality.png"), dpi=200)

    plt.figure(figsize=(7, 4))
    plt.plot(qs, rs, marker="o")
    plt.xlabel("Quality (1..100)")
    plt.ylabel("Compression ratio [x]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ratio_vs_quality.png"), dpi=200)

    plt.figure(figsize=(7, 4))
    plt.plot(rs, ps, marker="o")
    plt.xlabel("Compression ratio [x]")
    plt.ylabel("PSNR [dB]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "psnr_vs_ratio.png"), dpi=200)

    plt.figure(figsize=(7, 4))
    plt.plot(qs, fs, marker="o")
    plt.xlabel("Quality (1..100)")
    plt.ylabel("Decompression throughput [FPS]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fps_vs_quality.png"), dpi=200)

    print("\nSaved figures to:", out_dir)


if __name__ == "__main__":
    main()

