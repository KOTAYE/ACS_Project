#!/usr/bin/env python3
"""
Порівняння двох режимів flipbook_cuda:
  - FLI5 (за замовчуванням): різниця з prev у тій самій (x,y), без MV у бітстримі.
  - FLI6 + --motion-predict: блочний ME (SAD) + warp + MV у бітстримі.

Запускає компресію/декомпресію, парсить [BENCHMARK] з stderr, рахує середній PSNR
до оригінальних кадрів, будує bar-chart і зберігає JSON.

Залежності: numpy, pillow, matplotlib
  pip install numpy pillow matplotlib
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image
except ImportError as e:
    print("Потрібні пакети: pip install numpy pillow matplotlib", file=sys.stderr)
    raise SystemExit(1) from e

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print("Потрібен matplotlib: pip install matplotlib", file=sys.stderr)
    raise SystemExit(1) from e


def frame_path_numeric_key(name: str) -> list[int]:
    """Те саме правило, що `frame_path_numeric_key` у image_io.cpp (для порядку кадрів)."""
    nums: list[int] = []
    i = 0
    while i < len(name):
        if name[i].isdigit():
            v = 0
            while i < len(name) and name[i].isdigit():
                v = v * 10 + ord(name[i]) - 48
                i += 1
            nums.append(v)
        else:
            i += 1
    return nums


def sorted_frames(frames_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]

    def sort_key(p: Path) -> tuple:
        na = p.name
        ka = frame_path_numeric_key(na)
        if ka:
            return (0, tuple(ka), na.lower())
        return (1, (), na.lower())

    return sorted(paths, key=sort_key)


def run_flipbook(
    exe: Path,
    mode: str,
    *args: str,
    timeout: float | None = None,
) -> tuple[int, str]:
    cmd = [str(exe), mode, *args]
    p = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out


def parse_compress(text: str) -> dict[str, float]:
    m = re.search(
        r"compress_ms=([0-9.]+).*?raw_bytes=([0-9]+).*?compressed_bytes=([0-9]+).*?compression_ratio=([0-9.]+)",
        text,
        re.DOTALL,
    )
    if not m:
        return {}
    return {
        "compress_ms": float(m.group(1)),
        "raw_bytes": float(m.group(2)),
        "compressed_bytes": float(m.group(3)),
        "compression_ratio": float(m.group(4)),
    }


def parse_decode(text: str) -> dict[str, float]:
    m = re.search(r"decode_total_ms=([0-9.]+).*?frames=([0-9]+).*?decode_fps=([0-9.]+)", text, re.DOTALL)
    if not m:
        return {}
    return {
        "decode_total_ms": float(m.group(1)),
        "decode_frames": float(m.group(2)),
        "decode_fps": float(m.group(3)),
    }


def load_rgb_u8(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def psnr_rgb(a: np.ndarray, b: np.ndarray) -> float:
    """a,b uint8 HWC RGB same shape."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def mean_psnr_vs_sources(
    sources: list[Path],
    decoded_dir: Path,
    n_frames: int,
) -> float | None:
    """PSNR RGB vs вихідні файли. None, якщо кадри майже константні (PSNR неінформативний)."""
    for src in sources[: min(4, len(sources))]:
        try:
            if float(np.std(load_rgb_u8(src))) < 2.0:
                return None
        except OSError:
            return None
    psnrs: list[float] = []
    for i in range(n_frames):
        src = sources[i] if i < len(sources) else None
        dec = decoded_dir / f"frame_{i:04d}.png"
        if not src or not src.is_file() or not dec.is_file():
            continue
        try:
            a = load_rgb_u8(src)
            b = load_rgb_u8(dec)
            if a.shape != b.shape:
                h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
                a, b = a[:h, :w], b[:h, :w]
            psnrs.append(psnr_rgb(a, b))
        except OSError:
            continue
    if not psnrs:
        return None
    return float(np.mean(psnrs))


def main() -> None:
    ap = argparse.ArgumentParser(description="Порівняти FLI5 vs FLI6+motion (flipbook_cuda).")
    ap.add_argument("--frames-dir", type=Path, required=True, help="Каталог вхідних кадрів (jpeg/png)")
    ap.add_argument(
        "--exe",
        type=Path,
        default=None,
        help="Шлях до flipbook_cuda.exe (за замовчуванням: build/Debug або build/Release поруч із репо)",
    )
    ap.add_argument("-q", "--quality", type=int, default=50)
    ap.add_argument("-b", "--block-size", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=None, help="Куди зберегти графік і JSON (типово temp)")
    ap.add_argument("--timeout", type=float, default=None, help="Таймаут секунд на один запуск")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    exe = args.exe
    if exe is None:
        for cand in (
            repo / "build" / "Debug" / "flipbook_cuda.exe",
            repo / "build" / "Release" / "flipbook_cuda.exe",
            repo / "build" / "flipbook_cuda.exe",
        ):
            if cand.is_file():
                exe = cand
                break
        if exe is None:
            print("Не знайдено flipbook_cuda.exe. Вкажи --exe", file=sys.stderr)
            raise SystemExit(2)

    frames_dir: Path = args.frames_dir.resolve()
    if not frames_dir.is_dir():
        print(f"Немає каталогу: {frames_dir}", file=sys.stderr)
        raise SystemExit(2)

    sources = sorted_frames(frames_dir)
    if not sources:
        print("У каталозі немає кадрів (png/jpeg/…)", file=sys.stderr)
        raise SystemExit(2)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="flipbook_compare_"))
    else:
        out_dir = out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    common = [
        "-q",
        str(args.quality),
        "-b",
        str(args.block_size),
        str(frames_dir),
    ]

    results: dict[str, Any] = {
        "frames_dir": str(frames_dir),
        "exe": str(exe),
        "quality": args.quality,
        "block_size": args.block_size,
        "modes": {},
    }

    modes = [
        ("FLI5 (pixel delta)", "baseline.bin", []),
        ("FLI6 + motion", "motion.bin", ["--motion-predict"]),
    ]

    for label, bin_name, extra in modes:
        bin_path = out_dir / bin_name
        dec_dir = out_dir / f"dec_{bin_path.stem}"
        if dec_dir.exists():
            shutil.rmtree(dec_dir)
        dec_dir.mkdir()

        t0 = time.perf_counter()
        code, log_c = run_flipbook(
            exe,
            "compress",
            *extra,
            *common,
            str(bin_path),
            timeout=args.timeout,
        )
        enc_wall_s = time.perf_counter() - t0
        if code != 0:
            print(f"[FAIL] compress {label}\n{log_c}", file=sys.stderr)
            results["modes"][label] = {"error": code, "log": log_c[-4000:]}
            continue

        pc = parse_compress(log_c)
        sz = bin_path.stat().st_size if bin_path.is_file() else 0
        raw_uc = int(pc.get("raw_bytes", 0)) if pc else 0
        n_frames = len(sources)

        t1 = time.perf_counter()
        code_d, log_d = run_flipbook(
            exe,
            "decompress",
            str(bin_path),
            str(dec_dir),
            timeout=args.timeout,
        )
        dec_wall_s = time.perf_counter() - t1
        if code_d != 0:
            print(f"[FAIL] decompress {label}\n{log_d}", file=sys.stderr)
            results["modes"][label] = {"error_decode": code_d, "compress": pc, "log": log_d[-4000:]}
            continue

        pd = parse_decode(log_d)
        mpsnr = mean_psnr_vs_sources(sources, dec_dir, int(pd.get("decode_frames", n_frames)))
        psnr_note = None
        if mpsnr is None:
            psnr_note = "skipped_low_variance_sources"

        results["modes"][label] = {
            "compressed_bytes": int(sz),
            "raw_uncompressed_bytes": raw_uc,
            "compression_ratio_raw_over_bin": (raw_uc / sz) if sz and raw_uc else None,
            "compress_benchmark": pc,
            "compress_wall_s": enc_wall_s,
            "decode_benchmark": pd,
            "decode_wall_s": dec_wall_s,
            "mean_psnr_vs_sources": mpsnr,
            "psnr_note": psnr_note,
        }

    json_path = out_dir / "compare_flipbook_modes.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_analysis_txt() -> None:
        lines = [
            "Порівняння режимів flipbook_cuda",
            "==============================",
            "",
            "FLI5 (за замовчуванням): міжкадрова різниця prev[x,y] без motion vectors.",
            "FLI6 + --motion-predict: блочний SAD ME + warp + MV у бітстримі.",
            "",
        ]
        m5 = results["modes"].get("FLI5 (pixel delta)", {})
        m6 = results["modes"].get("FLI6 + motion", {})
        if isinstance(m5, dict) and isinstance(m6, dict) and "compressed_bytes" in m5 and "compressed_bytes" in m6:
            smaller = "FLI5" if m5["compressed_bytes"] <= m6["compressed_bytes"] else "FLI6+motion"
            lines.append(f"Розмір .bin: менший у {smaller} (краще стиснення за розміром файлу).")
            lines.append(f"  FLI5: {m5['compressed_bytes']} B  |  FLI6: {m6['compressed_bytes']} B")
            lines.append("")
        c5, c6 = m5.get("compress_benchmark", {}), m6.get("compress_benchmark", {})
        if c5 and c6:
            faster = "FLI5" if c5.get("compress_ms", 1e9) <= c6.get("compress_ms", 1e9) else "FLI6+motion"
            lines.append(f"Час compress (лог): швидший {faster} (нижчі ms — краще для продуктивності).")
            lines.append(f"  FLI5: {c5.get('compress_ms')} ms  |  FLI6: {c6.get('compress_ms')} ms")
            lines.append("")
        d5, d6 = m5.get("decode_benchmark", {}), m6.get("decode_benchmark", {})
        if d5 and d6:
            faster_d = "FLI5" if d5.get("decode_total_ms", 1e9) <= d6.get("decode_total_ms", 1e9) else "FLI6+motion"
            lines.append(f"Час decompress (лог): швидший {faster_d}.")
            lines.append(f"  FLI5: {d5.get('decode_total_ms')} ms  |  FLI6: {d6.get('decode_total_ms')} ms")
            lines.append("")
        p5, p6 = m5.get("mean_psnr_vs_sources"), m6.get("mean_psnr_vs_sources")
        if p5 is not None and p6 is not None and not (np.isnan(p5) and np.isnan(p6)):
            better = "FLI5" if p5 >= p6 else "FLI6+motion"
            lines.append(f"Середній PSNR до вихідних кадрів: вищий у {better} (ближче до оригіналу після round-trip).")
            lines.append(f"  FLI5: {p5:.2f} dB  |  FLI6: {p6:.2f} dB")
            lines.append("")
        elif m5.get("psnr_note") or m6.get("psnr_note"):
            lines.append("PSNR: пропущено (вихідні кадри майже без варіації яскравості — метрика неінформативна).")
            lines.append("")
        lines.append(
            "Примітка: PSNR — RGB з файлів кадрів vs PNG після decompress (квантування дає відхилення від оригіналу)."
        )
        (out_dir / "compare_flipbook_analysis.txt").write_text("\n".join(lines), encoding="utf-8")

    write_analysis_txt()

    # Графіки: лише якщо ≥2 успішні режими
    labels: list[str] = []
    ratio: list[float] = []
    enc_ms: list[float] = []
    dec_ms: list[float] = []
    psnr_v: list[float] = []
    bytes_mb: list[float] = []

    for label, _, _ in modes:
        block = results["modes"].get(label)
        if not block or "error" in block or "error_decode" in block:
            continue
        labels.append(label.replace(" ", "\n"))
        r = block.get("compress_benchmark", {}).get("compression_ratio")
        ratio.append(float(r) if r is not None else 0.0)
        enc_ms.append(float(block.get("compress_benchmark", {}).get("compress_ms", 0.0)))
        dec_ms.append(float(block.get("decode_benchmark", {}).get("decode_total_ms", 0.0)))
        p = block.get("mean_psnr_vs_sources")
        psnr_v.append(float(p) if p is not None else float("nan"))
        cb = block.get("compressed_bytes", 0)
        bytes_mb.append(float(cb) / (1024 * 1024))

    if len(labels) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("FLI5 vs FLI6+motion: порівняння flipbook_cuda", fontsize=12)

        x = np.arange(len(labels))
        colors = ["#4472c4", "#ed7d31", "#70ad47", "#c55e5e"]

        ax = axes[0, 0]
        ax.bar(x, ratio, color=colors[: len(x)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("compression_ratio (raw/bin)")
        ax.set_title("Стиснення (вище — краще)")

        ax = axes[0, 1]
        ax.bar(x, bytes_mb, color=colors[: len(x)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Розмір .bin (МБ)")
        ax.set_title("Розмір файлу (менше — краще)")

        ax = axes[1, 0]
        ax.bar(x, enc_ms, color=colors[: len(x)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("ms")
        ax.set_title("Час compress (з логу; нижче — швидше)")

        ax = axes[1, 1]
        ax.bar(x, dec_ms, color=colors[: len(x)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("ms")
        ax.set_title("Час decompress (з логу; нижче — швидше)")

        fig.tight_layout()
        fig.savefig(out_dir / "compare_flipbook_modes.png", dpi=150)
        plt.close(fig)

        if not all(np.isnan(psnr_v)):
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.bar(x, [v if not np.isnan(v) else 0 for v in psnr_v], color=colors[: len(x)])
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, fontsize=9)
            ax2.set_ylabel("Mean PSNR (dB)")
            ax2.set_title("Якість vs оригінальні кадри (вище — ближче до lossless)")
            fig2.tight_layout()
            fig2.savefig(out_dir / "compare_flipbook_psnr.png", dpi=150)
            plt.close(fig2)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nЗбережено: {json_path}")
    if (out_dir / "compare_flipbook_modes.png").is_file():
        print(f"Графік: {out_dir / 'compare_flipbook_modes.png'}")
    if (out_dir / "compare_flipbook_psnr.png").is_file():
        print(f"PSNR:   {out_dir / 'compare_flipbook_psnr.png'}")
    if (out_dir / "compare_flipbook_analysis.txt").is_file():
        print(f"Текст: {out_dir / 'compare_flipbook_analysis.txt'}")


if __name__ == "__main__":
    main()
