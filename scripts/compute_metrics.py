#!/usr/bin/env python3
"""Compute PSNR and SSIM between original frames and reconstructed frames."""
import sys, os, json, glob, re
import numpy as np


def image_sort_key(path):
    """Sort by numeric runs in filename so frame_2.png comes before frame_10.png."""
    name = os.path.basename(path)
    nums = [int(x) for x in re.findall(r"\d+", name)]
    return tuple(nums) if nums else (name,)

def load_image(path):
    try:
        from PIL import Image
        return np.array(Image.open(path))
    except ImportError:
        import subprocess
        # Fallback: use stb via a tiny C program, or just skip
        raise

def compute_psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 10.0 * np.log10(255.0 ** 2 / mse)

def compute_ssim_channel(a, b, window=8):
    K1, K2, L = 0.01, 0.03, 255.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    
    h, w = a.shape
    ssim_vals = []
    for y in range(0, h - window + 1, window):
        for x in range(0, w - window + 1, window):
            wa = a[y:y+window, x:x+window]
            wb = b[y:y+window, x:x+window]
            mu_a = wa.mean()
            mu_b = wb.mean()
            sigma_a2 = wa.var()
            sigma_b2 = wb.var()
            sigma_ab = np.mean((wa - mu_a) * (wb - mu_b))
            num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
            den = (mu_a**2 + mu_b**2 + C1) * (sigma_a2 + sigma_b2 + C2)
            ssim_vals.append(num / den)
    return np.mean(ssim_vals) if ssim_vals else 1.0

def compute_ssim(a, b):
    if a.ndim == 2:
        return compute_ssim_channel(a, b)
    vals = []
    for c in range(a.shape[2]):
        vals.append(compute_ssim_channel(a[:,:,c], b[:,:,c]))
    return np.mean(vals)

def parse_benchmark_log(log_path):
    result = {}
    if not os.path.exists(log_path):
        return result
    with open(log_path) as f:
        for line in f:
            m = re.search(r'\[BENCHMARK\]\s+(.*)', line)
            if m:
                for kv in m.group(1).split():
                    k, v = kv.split('=')
                    try:
                        result[k] = float(v)
                    except ValueError:
                        result[k] = v
    return result

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <original_frames_dir> <results_dir>")
        sys.exit(1)
    
    orig_dir = sys.argv[1]
    results_dir = sys.argv[2]
    
    orig_frames = sorted(
        glob.glob(os.path.join(orig_dir, "*.png"))
        + glob.glob(os.path.join(orig_dir, "*.jpg"))
        + glob.glob(os.path.join(orig_dir, "*.exr")),
        key=image_sort_key,
    )
    
    if not orig_frames:
        print(f"No frames found in {orig_dir}")
        sys.exit(1)
    
    print(f"Found {len(orig_frames)} original frames")
    
    all_results = {}
    
    for impl in ["gpu", "cpu"]:
        for q in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            key = f"{impl}_q{q}"
            recon_dir = os.path.join(results_dir, f"{key}_recon")
            
            # Parse benchmark logs
            compress_log = parse_benchmark_log(os.path.join(results_dir, f"{impl}_compress_q{q}.log"))
            decompress_log = parse_benchmark_log(os.path.join(results_dir, f"{impl}_decompress_q{q}.log"))
            
            entry = {
                "impl": impl,
                "quality": q,
                **compress_log,
                **decompress_log,
            }
            
            if os.path.isdir(recon_dir):
                recon_frames = sorted(
                    glob.glob(os.path.join(recon_dir, "*.png")), key=image_sort_key
                )

                psnr_vals = []
                ssim_vals = []
                if len(orig_frames) != len(recon_frames):
                    print(
                        f"  Warning {key}: orig {len(orig_frames)} vs recon {len(recon_frames)} frames"
                    )
                n = min(len(orig_frames), len(recon_frames))
                
                for i in range(n):
                    try:
                        orig_img = load_image(orig_frames[i])
                        recon_img = load_image(recon_frames[i])
                        
                        # Resize if needed
                        if orig_img.shape != recon_img.shape:
                            min_h = min(orig_img.shape[0], recon_img.shape[0])
                            min_w = min(orig_img.shape[1], recon_img.shape[1])
                            orig_img = orig_img[:min_h, :min_w]
                            recon_img = recon_img[:min_h, :min_w]
                        
                        psnr_vals.append(compute_psnr(orig_img, recon_img))
                        ssim_vals.append(compute_ssim(orig_img, recon_img))
                    except Exception as e:
                        print(f"  Warning: failed to process frame {i}: {e}")
                
                if psnr_vals:
                    entry["avg_psnr"] = float(np.mean(psnr_vals))
                    entry["avg_ssim"] = float(np.mean(ssim_vals))
                    entry["min_psnr"] = float(np.min(psnr_vals))
                    entry["max_psnr"] = float(np.max(psnr_vals))
                    print(f"  {key}: PSNR={entry['avg_psnr']:.2f} dB, SSIM={entry['avg_ssim']:.4f}")
            
            all_results[key] = entry
    
    out_path = os.path.join(results_dir, "metrics.json")
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved to {out_path}")

if __name__ == "__main__":
    main()
