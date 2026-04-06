#!/usr/bin/env python3
"""Generate publication-quality comparison charts from real benchmark data."""
import sys, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.8,
    'font.family': 'sans-serif',
    'font.size': 11,
})

GPU_COLOR = '#58a6ff'
CPU_COLOR = '#f0883e'
ACCENT1 = '#3fb950'
ACCENT2 = '#bc8cff'
ACCENT3 = '#ff7b72'

def load_metrics(results_dir):
    path = os.path.join(results_dir, "metrics.json")
    if not os.path.exists(path):
        print(f"Error: {path} not found. Run compute_metrics.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)

def get_series(data, impl, field):
    qualities = []
    values = []
    for q in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        key = f"{impl}_q{q}"
        if key in data and field in data[key]:
            qualities.append(q)
            values.append(data[key][field])
    return qualities, values

def chart1_psnr_vs_quality(data, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gq, gv = get_series(data, "gpu", "avg_psnr")
    cq, cv = get_series(data, "cpu", "avg_psnr")
    
    if gq:
        ax.plot(gq, gv, 'o-', color=GPU_COLOR, linewidth=2.5, markersize=8, label='CUDA (GPU)', zorder=5)
    if cq:
        ax.plot(cq, cv, 's--', color=CPU_COLOR, linewidth=2.5, markersize=8, label='CPU (OpenMP)', zorder=5)
    
    ax.axhline(y=20, color=ACCENT3, linestyle=':', alpha=0.7, label='Min target (20 dB)')
    ax.axhline(y=30, color=ACCENT1, linestyle=':', alpha=0.7, label='Good quality (30 dB)')
    
    ax.set_xlabel('Quality Parameter', fontsize=13)
    ax.set_ylabel('PSNR (dB)', fontsize=13)
    ax.set_title('PSNR vs Quality — GPU vs CPU', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', facecolor='#161b22', edgecolor='#30363d')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(5, 105)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'chart_psnr_quality.png'), dpi=150)
    plt.close()
    print("  Generated: chart_psnr_quality.png")

def chart2_ssim_vs_quality(data, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gq, gv = get_series(data, "gpu", "avg_ssim")
    cq, cv = get_series(data, "cpu", "avg_ssim")
    
    if gq:
        ax.plot(gq, gv, 'o-', color=GPU_COLOR, linewidth=2.5, markersize=8, label='CUDA (GPU)')
    if cq:
        ax.plot(cq, cv, 's--', color=CPU_COLOR, linewidth=2.5, markersize=8, label='CPU (OpenMP)')
    
    ax.axhline(y=0.9, color=ACCENT1, linestyle=':', alpha=0.7, label='Good quality (0.9)')
    
    ax.set_xlabel('Quality Parameter', fontsize=13)
    ax.set_ylabel('SSIM', fontsize=13)
    ax.set_title('SSIM vs Quality — GPU vs CPU', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', facecolor='#161b22', edgecolor='#30363d')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(5, 105)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'chart_ssim_quality.png'), dpi=150)
    plt.close()
    print("  Generated: chart_ssim_quality.png")

def chart3_compression_ratio(data, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gq, gv = get_series(data, "gpu", "compression_ratio")
    cq, cv = get_series(data, "cpu", "compression_ratio")
    
    if gq:
        ax.plot(gq, gv, 'o-', color=GPU_COLOR, linewidth=2.5, markersize=8, label='CUDA (GPU)')
        for x, y in zip(gq, gv):
            ax.annotate(f'{y:.1f}x', (x, y), textcoords="offset points",
                       xytext=(0, 12), ha='center', fontsize=9, color=GPU_COLOR)
    if cq:
        ax.plot(cq, cv, 's--', color=CPU_COLOR, linewidth=2.5, markersize=8, label='CPU (OpenMP)')
        for x, y in zip(cq, cv):
            ax.annotate(f'{y:.1f}x', (x, y), textcoords="offset points",
                       xytext=(0, -18), ha='center', fontsize=9, color=CPU_COLOR)
    
    ax.axhline(y=10, color=ACCENT1, linestyle=':', alpha=0.7, label='Target (10x)')
    
    ax.set_xlabel('Quality Parameter', fontsize=13)
    ax.set_ylabel('Compression Ratio', fontsize=13)
    ax.set_title('Compression Ratio vs Quality', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', facecolor='#161b22', edgecolor='#30363d')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(5, 105)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'chart_compression_ratio.png'), dpi=150)
    plt.close()
    print("  Generated: chart_compression_ratio.png")

def chart4_decode_fps(data, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gq, gv = get_series(data, "gpu", "decode_fps")
    cq, cv = get_series(data, "cpu", "decode_fps")
    
    if gq:
        ax.plot(gq, gv, 'o-', color=GPU_COLOR, linewidth=2.5, markersize=8, label='CUDA (GPU)')
    if cq:
        ax.plot(cq, cv, 's--', color=CPU_COLOR, linewidth=2.5, markersize=8, label='CPU (OpenMP)')
    
    ax.axhline(y=60, color=ACCENT1, linestyle=':', alpha=0.7, label='Target (60 FPS)')
    ax.axhline(y=30, color=ACCENT3, linestyle=':', alpha=0.7, label='Min playback (30 FPS)')
    
    ax.set_xlabel('Quality Parameter', fontsize=13)
    ax.set_ylabel('Decode FPS', fontsize=13)
    ax.set_title('Decompression Speed — GPU vs CPU', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', facecolor='#161b22', edgecolor='#30363d')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(5, 105)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'chart_decode_fps.png'), dpi=150)
    plt.close()
    print("  Generated: chart_decode_fps.png")

def chart5_compress_fps(data, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gq, gv = get_series(data, "gpu", "compress_fps")
    cq, cv = get_series(data, "cpu", "compress_fps")
    
    if gq:
        ax.plot(gq, gv, 'o-', color=GPU_COLOR, linewidth=2.5, markersize=8, label='CUDA (GPU)')
    if cq:
        ax.plot(cq, cv, 's--', color=CPU_COLOR, linewidth=2.5, markersize=8, label='CPU (OpenMP)')
    
    ax.set_xlabel('Quality Parameter', fontsize=13)
    ax.set_ylabel('Compress FPS', fontsize=13)
    ax.set_title('Compression Speed — GPU vs CPU', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', facecolor='#161b22', edgecolor='#30363d')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(5, 105)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'chart_compress_fps.png'), dpi=150)
    plt.close()
    print("  Generated: chart_compress_fps.png")

def chart6_rate_distortion(data, out_dir):
    """Rate-distortion curve: bits per pixel vs PSNR"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for impl, color, marker, label in [("gpu", GPU_COLOR, 'o', 'CUDA (GPU)'),
                                         ("cpu", CPU_COLOR, 's', 'CPU (OpenMP)')]:
        bpp_vals = []
        psnr_vals = []
        for q in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            key = f"{impl}_q{q}"
            if key in data and "avg_psnr" in data[key] and "compressed_bytes" in data[key] and "raw_bytes" in data[key]:
                raw = data[key]["raw_bytes"]
                comp = data[key]["compressed_bytes"]
                # Assume 8 bits per pixel originally
                bpp = 8.0 * comp / raw
                bpp_vals.append(bpp)
                psnr_vals.append(data[key]["avg_psnr"])
        
        if bpp_vals:
            ax.plot(bpp_vals, psnr_vals, f'{marker}-', color=color, linewidth=2.5, markersize=8, label=label)
            for bpp, psnr in zip(bpp_vals, psnr_vals):
                ax.annotate(f'q={[10,20,30,40,50,60,70,80,90,100][[round(b,3) for b in bpp_vals].index(round(bpp,3))]}',
                           (bpp, psnr), textcoords="offset points", xytext=(8, 4),
                           fontsize=8, color=color, alpha=0.8)
    
    ax.set_xlabel('Bits Per Pixel (bpp)', fontsize=13)
    ax.set_ylabel('PSNR (dB)', fontsize=13)
    ax.set_title('Rate-Distortion Curve', fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', facecolor='#161b22', edgecolor='#30363d')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'chart_rate_distortion.png'), dpi=150)
    plt.close()
    print("  Generated: chart_rate_distortion.png")

def chart7_speedup(data, out_dir):
    """GPU speedup over CPU for compress and decompress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    qualities = []
    compress_speedup = []
    decompress_speedup = []
    
    for q in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        gk = f"gpu_q{q}"
        ck = f"cpu_q{q}"
        if gk in data and ck in data:
            if "compress_ms" in data[gk] and "compress_ms" in data[ck]:
                qualities.append(q)
                compress_speedup.append(data[ck]["compress_ms"] / max(data[gk]["compress_ms"], 0.001))
            if "decode_total_ms" in data[gk] and "decode_total_ms" in data[ck]:
                decompress_speedup.append(data[ck]["decode_total_ms"] / max(data[gk]["decode_total_ms"], 0.001))
    
    if qualities and compress_speedup:
        bars1 = ax1.bar(qualities, compress_speedup, width=7, color=GPU_COLOR, alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars1, compress_speedup):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}x', ha='center', va='bottom', fontsize=9, color=GPU_COLOR)
        ax1.axhline(y=1, color=ACCENT3, linestyle=':', alpha=0.7, label='1x (CPU baseline)')
        ax1.set_xlabel('Quality', fontsize=13)
        ax1.set_ylabel('Speedup (GPU / CPU)', fontsize=13)
        ax1.set_title('Compression Speedup', fontsize=14, fontweight='bold')
        ax1.legend(facecolor='#161b22', edgecolor='#30363d')
        ax1.grid(True, alpha=0.3, axis='y')
    
    if qualities and decompress_speedup:
        bars2 = ax2.bar(qualities, decompress_speedup, width=7, color=ACCENT1, alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars2, decompress_speedup):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}x', ha='center', va='bottom', fontsize=9, color=ACCENT1)
        ax2.axhline(y=1, color=ACCENT3, linestyle=':', alpha=0.7, label='1x (CPU baseline)')
        ax2.set_xlabel('Quality', fontsize=13)
        ax2.set_ylabel('Speedup (GPU / CPU)', fontsize=13)
        ax2.set_title('Decompression Speedup', fontsize=14, fontweight='bold')
        ax2.legend(facecolor='#161b22', edgecolor='#30363d')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'chart_speedup.png'), dpi=150)
    plt.close()
    print("  Generated: chart_speedup.png")

def chart8_summary_table(data, out_dir):
    """Summary comparison at quality=50"""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis('off')
    
    metrics = [
        ("Compression Ratio", "compression_ratio", "x", ".1f"),
        ("PSNR (dB)", "avg_psnr", " dB", ".2f"),
        ("SSIM", "avg_ssim", "", ".4f"),
        ("Compress FPS", "compress_fps", " fps", ".1f"),
        ("Decode FPS", "decode_fps", " fps", ".1f"),
        ("Compress Time (ms)", "compress_ms", " ms", ".1f"),
        ("Decode Time (ms)", "decode_total_ms", " ms", ".1f"),
    ]
    
    col_labels = ["Metric", "CUDA (GPU)", "CPU (OpenMP)", "Speedup"]
    table_data = []
    
    gpu_data = data.get("gpu_q50", {})
    cpu_data = data.get("cpu_q50", {})
    
    for name, field, suffix, fmt in metrics:
        gpu_val = gpu_data.get(field, None)
        cpu_val = cpu_data.get(field, None)
        
        gpu_str = f"{gpu_val:{fmt}}{suffix}" if gpu_val is not None else "N/A"
        cpu_str = f"{cpu_val:{fmt}}{suffix}" if cpu_val is not None else "N/A"
        
        if gpu_val and cpu_val and field in ["compress_ms", "decode_total_ms"]:
            speedup = f"{cpu_val / gpu_val:.1f}x"
        elif gpu_val and cpu_val and field in ["compress_fps", "decode_fps"]:
            speedup = f"{gpu_val / cpu_val:.1f}x"
        else:
            speedup = "—"
        
        table_data.append([name, gpu_str, cpu_str, speedup])
    
    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#30363d')
        if row == 0:
            cell.set_facecolor('#1f6feb')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#161b22')
        else:
            cell.set_facecolor('#0d1117')
        cell.set_text_props(color='#c9d1d9')
    
    ax.set_title('Performance Summary at Quality=50', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'chart_summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Generated: chart_summary_table.png")

def chart9_psnr_ssim_combined(data, out_dir):
    """Combined PSNR and SSIM on dual axes"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    gq_p, gv_p = get_series(data, "gpu", "avg_psnr")
    gq_s, gv_s = get_series(data, "gpu", "avg_ssim")
    
    if gq_p:
        l1, = ax1.plot(gq_p, gv_p, 'o-', color=GPU_COLOR, linewidth=2.5, markersize=8, label='GPU PSNR')
    if gq_s:
        l2, = ax2.plot(gq_s, gv_s, '^-', color=ACCENT1, linewidth=2.5, markersize=8, label='GPU SSIM')
    
    cq_p, cv_p = get_series(data, "cpu", "avg_psnr")
    cq_s, cv_s = get_series(data, "cpu", "avg_ssim")
    
    if cq_p:
        l3, = ax1.plot(cq_p, cv_p, 's--', color=CPU_COLOR, linewidth=2.5, markersize=8, label='CPU PSNR')
    if cq_s:
        l4, = ax2.plot(cq_s, cv_s, 'D--', color=ACCENT2, linewidth=2.5, markersize=8, label='CPU SSIM')
    
    ax1.set_xlabel('Quality Parameter', fontsize=13)
    ax1.set_ylabel('PSNR (dB)', fontsize=13, color=GPU_COLOR)
    ax2.set_ylabel('SSIM', fontsize=13, color=ACCENT1)
    ax1.set_title('Image Quality: PSNR & SSIM vs Quality', fontsize=15, fontweight='bold')
    
    lines = []
    labels = []
    for ax in [ax1, ax2]:
        for l, lab in zip(*ax.get_legend_handles_labels()):
            lines.append(l)
            labels.append(lab)
    ax1.legend(lines, labels, loc='lower right', facecolor='#161b22', edgecolor='#30363d')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(5, 105)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'chart_quality_combined.png'), dpi=150)
    plt.close()
    print("  Generated: chart_quality_combined.png")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    data = load_metrics(results_dir)
    
    out_dir = "charts"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Generating charts from benchmark data...")
    chart1_psnr_vs_quality(data, out_dir)
    chart2_ssim_vs_quality(data, out_dir)
    chart3_compression_ratio(data, out_dir)
    chart4_decode_fps(data, out_dir)
    chart5_compress_fps(data, out_dir)
    chart6_rate_distortion(data, out_dir)
    chart7_speedup(data, out_dir)
    chart8_summary_table(data, out_dir)
    chart9_psnr_ssim_combined(data, out_dir)
    print(f"\nAll charts saved to {out_dir}/")

if __name__ == "__main__":
    main()
