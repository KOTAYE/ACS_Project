#!/usr/bin/env python3
"""
Live camera compression using flipbook_cuda in chunked mode.

This script captures frames from a webcam and continuously compresses them
in short chunks. Target bitrate is specified in Mbps and translated into
per-chunk target size for the existing rate-control path (--target-size-mb).
"""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class ChunkJob:
    idx: int
    frame_dir: Path
    output_bin: Path
    frames_in_chunk: int
    fps: float
    target_mbps: float
    base_quality: int
    block_size: int
    scene_cut_threshold: float
    use_ycbcr: bool
    encoder_exe: Path
    chunk_start_ts: float
    chunk_end_ts: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live camera compression with target Mbps (chunked streaming)."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--fps", type=float, default=30.0, help="Capture FPS (default: 30)")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=1.0,
        help="Chunk duration in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--target-mbps",
        type=float,
        required=True,
        help="Target bitrate in Mbps for compressed stream",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=70,
        help="Base quality (1-100) for encoder (default: 70)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=8,
        choices=[8, 16, 32],
        help="DCT block size (default: 8)",
    )
    parser.add_argument(
        "--scene-cut-threshold",
        type=float,
        default=22.0,
        help="Mean luma diff threshold for auto keyframe (default: 22.0)",
    )
    parser.add_argument(
        "--no-ycbcr",
        action="store_true",
        help="Disable YCbCr conversion in encoder",
    )
    parser.add_argument(
        "--encoder-exe",
        default=str(Path("build") / "Release" / "flipbook_cuda.exe"),
        help="Path to flipbook_cuda executable",
    )
    parser.add_argument(
        "--output-dir",
        default="live_stream_output",
        help="Output directory for compressed chunks",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Stop after N seconds (0 = until Ctrl+C)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show live preview window",
    )
    return parser.parse_args()


def compress_chunk(job: ChunkJob) -> tuple[ChunkJob, int, str]:
    target_size_mb = (job.target_mbps * (job.frames_in_chunk / max(job.fps, 1e-3))) / 8.0
    cmd = [
        str(job.encoder_exe),
        "compress",
        "-q",
        str(job.base_quality),
        "-b",
        str(job.block_size),
        "--target-size-mb",
        f"{target_size_mb:.6f}",
        "--scene-cut-threshold",
        f"{job.scene_cut_threshold:.4f}",
    ]
    if not job.use_ycbcr:
        cmd.append("--no-ycbcr")
    cmd.extend([str(job.frame_dir), str(job.output_bin)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr_or_stdout = (result.stderr or "").strip()
    if not stderr_or_stdout:
        stderr_or_stdout = (result.stdout or "").strip()
    return job, result.returncode, stderr_or_stdout


def capture_worker(
    cap: cv2.VideoCapture,
    frame_queue: "queue.Queue[tuple[float, object]]",
    stop_event: threading.Event,
    preview: bool,
) -> None:
    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            ts = time.time()
            try:
                frame_queue.put((ts, frame), timeout=0.05)
            except queue.Full:
                # Drop frame when pipeline is overloaded to keep latency bounded.
                pass

            if preview:
                cv2.imshow("live_camera_compress preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break
    finally:
        if preview:
            cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()

    encoder_exe = Path(args.encoder_exe).resolve()
    if not encoder_exe.exists():
        print(f"[ERROR] Encoder not found: {encoder_exe}")
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.tsv"

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.camera}")
        return 1

    if args.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 1e-3:
        actual_fps = args.fps

    chunk_frames = max(1, int(round(actual_fps * args.chunk_seconds)))
    print(
        f"[INFO] camera={args.camera}, fps={actual_fps:.2f}, chunk_frames={chunk_frames}, "
        f"target_mbps={args.target_mbps:.3f}"
    )

    frame_queue: "queue.Queue[tuple[float, object]]" = queue.Queue(maxsize=256)
    stop_event = threading.Event()
    capture_thread = threading.Thread(
        target=capture_worker,
        args=(cap, frame_queue, stop_event, args.preview),
        daemon=True,
    )
    capture_thread.start()

    pool = ThreadPoolExecutor(max_workers=1)
    pending_jobs: list[Future] = []
    started_at = time.time()
    chunk_idx = 0
    current_chunk_dir: Path | None = None
    current_count = 0
    current_start_ts = 0.0
    current_end_ts = 0.0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        manifest.write(
            "chunk_idx\tstart_ts\tend_ts\tframes\ttarget_mbps\toutput_bin\tstatus\n"
        )
        try:
            while True:
                if args.max_seconds > 0 and (time.time() - started_at) >= args.max_seconds:
                    stop_event.set()

                if stop_event.is_set() and frame_queue.empty():
                    break

                try:
                    ts, frame = frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if current_chunk_dir is None:
                    current_chunk_dir = Path(tempfile.mkdtemp(prefix=f"live_chunk_{chunk_idx:06d}_"))
                    current_count = 0
                    current_start_ts = ts

                frame_name = current_chunk_dir / f"frame_{current_count:05d}.png"
                if not cv2.imwrite(str(frame_name), frame):
                    print(f"[WARN] Failed to write frame: {frame_name}")
                    continue
                current_count += 1
                current_end_ts = ts

                if current_count >= chunk_frames:
                    out_bin = chunks_dir / f"chunk_{chunk_idx:06d}.bin"
                    job = ChunkJob(
                        idx=chunk_idx,
                        frame_dir=current_chunk_dir,
                        output_bin=out_bin,
                        frames_in_chunk=current_count,
                        fps=actual_fps,
                        target_mbps=args.target_mbps,
                        base_quality=args.quality,
                        block_size=args.block_size,
                        scene_cut_threshold=args.scene_cut_threshold,
                        use_ycbcr=not args.no_ycbcr,
                        encoder_exe=encoder_exe,
                        chunk_start_ts=current_start_ts,
                        chunk_end_ts=current_end_ts,
                    )
                    pending_jobs.append(pool.submit(compress_chunk, job))
                    current_chunk_dir = None
                    chunk_idx += 1

                done_futures = [f for f in pending_jobs if f.done()]
                for fut in done_futures:
                    pending_jobs.remove(fut)
                    job, code, info = fut.result()
                    status = "ok" if code == 0 else f"fail({code})"
                    manifest.write(
                        f"{job.idx}\t{job.chunk_start_ts:.6f}\t{job.chunk_end_ts:.6f}\t"
                        f"{job.frames_in_chunk}\t{job.target_mbps:.6f}\t{job.output_bin.name}\t{status}\n"
                    )
                    manifest.flush()
                    if code != 0:
                        print(f"[ERROR] Chunk {job.idx} compression failed: {info}")
                    else:
                        size_mb = job.output_bin.stat().st_size / (1024.0 * 1024.0)
                        print(
                            f"[OK] chunk={job.idx} frames={job.frames_in_chunk} "
                            f"size={size_mb:.3f}MB file={job.output_bin.name}"
                        )
                    shutil.rmtree(job.frame_dir, ignore_errors=True)
        except KeyboardInterrupt:
            print("\n[INFO] Stopping...")
            stop_event.set()
        finally:
            stop_event.set()
            capture_thread.join(timeout=2.0)
            cap.release()
            if current_chunk_dir is not None and current_count > 0:
                out_bin = chunks_dir / f"chunk_{chunk_idx:06d}.bin"
                job = ChunkJob(
                    idx=chunk_idx,
                    frame_dir=current_chunk_dir,
                    output_bin=out_bin,
                    frames_in_chunk=current_count,
                    fps=actual_fps,
                    target_mbps=args.target_mbps,
                    base_quality=args.quality,
                    block_size=args.block_size,
                    scene_cut_threshold=args.scene_cut_threshold,
                    use_ycbcr=not args.no_ycbcr,
                    encoder_exe=encoder_exe,
                    chunk_start_ts=current_start_ts,
                    chunk_end_ts=current_end_ts,
                )
                pending_jobs.append(pool.submit(compress_chunk, job))

            for fut in pending_jobs:
                job, code, info = fut.result()
                status = "ok" if code == 0 else f"fail({code})"
                manifest.write(
                    f"{job.idx}\t{job.chunk_start_ts:.6f}\t{job.chunk_end_ts:.6f}\t"
                    f"{job.frames_in_chunk}\t{job.target_mbps:.6f}\t{job.output_bin.name}\t{status}\n"
                )
                manifest.flush()
                if code != 0:
                    print(f"[ERROR] Chunk {job.idx} compression failed: {info}")
                else:
                    size_mb = job.output_bin.stat().st_size / (1024.0 * 1024.0)
                    print(
                        f"[OK] chunk={job.idx} frames={job.frames_in_chunk} "
                        f"size={size_mb:.3f}MB file={job.output_bin.name}"
                    )
                shutil.rmtree(job.frame_dir, ignore_errors=True)

    pool.shutdown(wait=True)
    print(f"[DONE] Output: {output_dir}")
    print(f"[DONE] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
