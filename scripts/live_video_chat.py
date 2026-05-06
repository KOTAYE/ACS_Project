#!/usr/bin/env python3
"""
Live camera streaming with bitrate-controlled compressed transport.

Modes:
  - sender: captures webcam, JPEG-compresses frames, streams over TCP.
  - receiver: accepts TCP stream, decodes frames, displays video.

This is a practical real-time transport prototype (no intermediate .bin files).
"""

from __future__ import annotations

import argparse
import collections
import socket
import struct
import sys
import time
from typing import Deque, Tuple

import cv2
import numpy as np


HEADER_STRUCT = struct.Struct("!dI")  # timestamp_sec(double), payload_len(uint32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live compressed video sender/receiver over TCP.")
    sub = parser.add_subparsers(dest="mode", required=True)

    send = sub.add_parser("sender", help="Capture camera and send compressed stream.")
    send.add_argument("--host", default="127.0.0.1", help="Receiver host/IP.")
    send.add_argument("--port", type=int, default=5000, help="Receiver TCP port.")
    send.add_argument("--camera", type=int, default=0, help="Camera index.")
    send.add_argument("--fps", type=float, default=30.0, help="Target capture FPS.")
    send.add_argument(
        "--max-fps",
        action="store_true",
        help="Run sender loop without FPS throttling (maximum throughput).",
    )
    send.add_argument("--width", type=int, default=1280, help="Capture width.")
    send.add_argument("--height", type=int, default=720, help="Capture height.")
    send.add_argument("--target-mbps", type=float, default=3.0, help="Target network bitrate in Mbps.")
    send.add_argument("--jpeg-quality", type=int, default=70, help="Initial JPEG quality (1..100).")
    send.add_argument("--min-quality", type=int, default=20, help="Minimum JPEG quality.")
    send.add_argument("--max-quality", type=int, default=90, help="Maximum JPEG quality.")
    send.add_argument("--preview", action="store_true", help="Show local sender preview.")
    send.add_argument(
        "--quality-slider",
        action="store_true",
        help="Enable live quality slider in preview window (manual quality control).",
    )

    recv = sub.add_parser("receiver", help="Receive compressed stream and display video.")
    recv.add_argument("--bind", default="0.0.0.0", help="Bind interface.")
    recv.add_argument("--port", type=int, default=5000, help="Listen TCP port.")
    recv.add_argument("--show-latency", action="store_true", help="Overlay one-way latency estimate.")

    return parser.parse_args()


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < n:
        data = sock.recv(n - len(chunks))
        if not data:
            raise ConnectionError("Connection closed by peer")
        chunks.extend(data)
    return bytes(chunks)


def sender_mode(args: argparse.Namespace) -> int:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.camera}")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    quality = max(1, min(100, args.jpeg_quality))
    min_q = max(1, min(100, args.min_quality))
    max_q = max(1, min(100, args.max_quality))
    if min_q > max_q:
        min_q, max_q = max_q, min_q

    send_window: Deque[Tuple[float, int]] = collections.deque()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[INFO] Connecting to {args.host}:{args.port} ...")
    sock.connect((args.host, args.port))
    print("[INFO] Connected. Press Ctrl+C to stop.")
    if args.quality_slider and not args.preview:
        print("[WARN] --quality-slider requires --preview; enabling preview.")
        args.preview = True

    window_name = "live_video_chat sender"
    if args.preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if args.quality_slider and args.preview:
        cv2.createTrackbar("Quality", window_name, int(quality), 100, lambda _x: None)
        cv2.setTrackbarMin("Quality", window_name, int(min_q))

    frame_interval = 1.0 / max(args.fps, 1.0)
    frame_idx = 0
    last_stats_t = time.time()
    fps_window: Deque[float] = collections.deque()

    try:
        while True:
            tick = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue

            if args.quality_slider and args.preview:
                slider_q = cv2.getTrackbarPos("Quality", window_name)
                slider_q = max(min_q, min(max_q, int(slider_q)))
                quality = slider_q

            enc_ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            if not enc_ok:
                continue
            payload = enc.tobytes()

            ts = time.time()
            packet = HEADER_STRUCT.pack(ts, len(payload)) + payload
            sock.sendall(packet)

            send_window.append((ts, len(packet)))
            cutoff = ts - 1.0
            while send_window and send_window[0][0] < cutoff:
                send_window.popleft()

            bytes_last_sec = sum(sz for _, sz in send_window)
            mbps_now = (bytes_last_sec * 8.0) / 1_000_000.0

            # Bitrate feedback control on quality.
            if frame_idx % 10 == 0 and not args.quality_slider:
                if mbps_now > args.target_mbps * 1.05 and quality > min_q:
                    quality = max(min_q, quality - 2)
                elif mbps_now < args.target_mbps * 0.80 and quality < max_q:
                    quality = min(max_q, quality + 1)

            if args.preview:
                preview = frame.copy()
                fps_window.append(ts)
                fps_cutoff = ts - 1.0
                while fps_window and fps_window[0] < fps_cutoff:
                    fps_window.popleft()
                fps_now = float(len(fps_window))
                cv2.putText(
                    preview,
                    f"TX {mbps_now:.2f} Mbps | q={quality} | fps~{fps_now:.0f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now = time.time()
            if now - last_stats_t >= 1.0:
                fps_now = float(len(fps_window)) if args.preview else 0.0
                print(f"[TX] bitrate={mbps_now:.2f} Mbps, q={quality}, fps~{fps_now:.0f}, frame={frame_idx}")
                last_stats_t = now

            frame_idx += 1
            if not args.max_fps:
                dt = time.time() - tick
                if dt < frame_interval:
                    time.sleep(frame_interval - dt)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        sock.close()
        cv2.destroyAllWindows()
    return 0


def receiver_mode(args: argparse.Namespace) -> int:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.port))
    srv.listen(1)
    print(f"[INFO] Listening on {args.bind}:{args.port} ...")
    conn, addr = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[INFO] Client connected: {addr[0]}:{addr[1]}")

    last_stats_t = time.time()
    rx_frames = 0
    rx_bytes = 0

    try:
        while True:
            hdr = recv_exact(conn, HEADER_STRUCT.size)
            sent_ts, payload_len = HEADER_STRUCT.unpack(hdr)
            payload = recv_exact(conn, payload_len)
            rx_bytes += HEADER_STRUCT.size + payload_len
            rx_frames += 1

            arr = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            if args.show_latency:
                latency_ms = (time.time() - sent_ts) * 1000.0
                cv2.putText(
                    frame,
                    f"Latency ~ {latency_ms:.1f} ms",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("live_video_chat receiver", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            now = time.time()
            if now - last_stats_t >= 1.0:
                mbps = (rx_bytes * 8.0) / ((now - last_stats_t) * 1_000_000.0)
                print(f"[RX] bitrate={mbps:.2f} Mbps, fps={rx_frames}")
                rx_frames = 0
                rx_bytes = 0
                last_stats_t = now
    except (KeyboardInterrupt, ConnectionError):
        pass
    finally:
        conn.close()
        srv.close()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "sender":
        return sender_mode(args)
    if args.mode == "receiver":
        return receiver_mode(args)
    print("Unknown mode")
    return 1


if __name__ == "__main__":
    sys.exit(main())
