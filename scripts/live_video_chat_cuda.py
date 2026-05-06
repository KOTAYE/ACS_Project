#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import ctypes
import select
import socket
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np


NET_HDR = struct.Struct("!dI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live chat over TCP using in-memory CUDA codec API.")
    sub = parser.add_subparsers(dest="mode", required=True)

    snd = sub.add_parser("sender", help="Capture camera and send in-memory CUDA packets.")
    snd.add_argument("--host", default="127.0.0.1")
    snd.add_argument("--port", type=int, default=6000)
    snd.add_argument("--camera", type=int, default=0)
    snd.add_argument("--fps", type=float, default=30.0)
    snd.add_argument("--max-fps", action="store_true")
    snd.add_argument("--width", type=int, default=640)
    snd.add_argument("--height", type=int, default=360)
    snd.add_argument("--quality", type=int, default=70)
    snd.add_argument("--min-quality", type=int, default=10)
    snd.add_argument("--max-quality", type=int, default=95)
    snd.add_argument("--target-mbps", type=float, default=2.0)
    snd.add_argument("--block-size", type=int, default=8, choices=[8, 16, 32])
    snd.add_argument("--scene-cut-threshold", type=float, default=22.0)
    snd.add_argument("--adaptive-roi", action="store_true")
    snd.add_argument("--roi-strength", type=float, default=0.55)
    snd.add_argument("--preview", action="store_true")
    snd.add_argument("--quality-slider", action="store_true")
    snd.add_argument("--codec-dll", default=str(Path("build") / "Release" / "realtime_codec.dll"))

    rcv = sub.add_parser("receiver", help="Receive in-memory CUDA packets and display.")
    rcv.add_argument("--bind", default="0.0.0.0")
    rcv.add_argument("--port", type=int, default=6000)
    rcv.add_argument("--show-latency", action="store_true")
    rcv.add_argument("--codec-dll", default=str(Path("build") / "Release" / "realtime_codec.dll"))
    return parser.parse_args()


def recv_exact(sock: socket.socket, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed by peer")
        data.extend(chunk)
    return bytes(data)


def load_codec_api(dll_path: str):
    dll = ctypes.CDLL(str(Path(dll_path).resolve()))

    dll.rtc_encoder_create.restype = ctypes.c_void_p
    dll.rtc_encoder_destroy.argtypes = [ctypes.c_void_p]
    dll.rtc_encoder_init.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float,
    ]
    dll.rtc_encoder_init.restype = ctypes.c_int
    dll.rtc_encoder_set_quality.argtypes = [ctypes.c_void_p, ctypes.c_int]
    dll.rtc_encoder_encode_packet.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
    ]
    dll.rtc_encoder_encode_packet.restype = ctypes.c_int

    dll.rtc_decoder_create.restype = ctypes.c_void_p
    dll.rtc_decoder_destroy.argtypes = [ctypes.c_void_p]
    dll.rtc_decoder_decode_packet.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    dll.rtc_decoder_decode_packet.restype = ctypes.c_int
    dll.rtc_free_buffer.argtypes = [ctypes.c_void_p]
    return dll


def sender_mode(args: argparse.Namespace) -> int:
    api = load_codec_api(args.codec_dll)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] cannot open camera index {args.camera}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    q = max(1, min(100, int(args.quality)))
    q_min = max(1, min(100, int(args.min_quality)))
    q_max = max(1, min(100, int(args.max_quality)))
    if q_min > q_max:
        q_min, q_max = q_max, q_min

    ok, first = cap.read()
    if not ok:
        print("[ERROR] failed to read first camera frame")
        return 1
    h, w = first.shape[:2]
    c = 3

    enc = api.rtc_encoder_create()
    if not enc:
        print("[ERROR] rtc_encoder_create failed")
        return 1
    init_ok = api.rtc_encoder_init(
        enc, w, h, c, q, int(args.block_size), 1,
        1 if args.adaptive_roi else 0, float(args.roi_strength), float(args.scene_cut_threshold)
    )
    if init_ok != 1:
        print("[ERROR] rtc_encoder_init failed")
        api.rtc_encoder_destroy(enc)
        return 1

    if args.quality_slider and not args.preview:
        print("[WARN] --quality-slider requires --preview; enabling preview.")
        args.preview = True
    wnd = "live_video_chat_cuda sender"
    if args.preview:
        cv2.namedWindow(wnd, cv2.WINDOW_NORMAL)
    if args.quality_slider and args.preview:
        cv2.createTrackbar("Quality", wnd, q, 100, lambda _x: None)
        cv2.setTrackbarMin("Quality", wnd, q_min)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.connect((args.host, args.port))
    print(f"[INFO] connected to {args.host}:{args.port}")

    frame_interval = 1.0 / max(args.fps, 1.0)
    tx_window = collections.deque()
    fps_window = collections.deque()
    frame_idx = 0
    last_log = time.time()

    def process_frame(frame: np.ndarray) -> bool:
        nonlocal q, frame_idx, last_log
        if args.quality_slider and args.preview:
            q = max(q_min, min(q_max, cv2.getTrackbarPos("Quality", wnd)))
        api.rtc_encoder_set_quality(enc, int(q))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        arr = np.ascontiguousarray(rgb_frame)
        raw = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        out_ptr = ctypes.c_void_p()
        out_sz = ctypes.c_int(0)
        if api.rtc_encoder_encode_packet(enc, raw, arr.size, ctypes.byref(out_ptr), ctypes.byref(out_sz)) != 1:
            return False
        try:
            payload = ctypes.string_at(out_ptr.value, out_sz.value)
        finally:
            api.rtc_free_buffer(out_ptr)

        ts = time.time()
        sock.sendall(NET_HDR.pack(ts, len(payload)) + payload)
        tx_window.append((ts, len(payload) + NET_HDR.size))
        fps_window.append(ts)
        cutoff = ts - 1.0
        while tx_window and tx_window[0][0] < cutoff:
            tx_window.popleft()
        while fps_window and fps_window[0] < cutoff:
            fps_window.popleft()
        mbps_now = (sum(b for _, b in tx_window) * 8.0) / 1_000_000.0
        fps_now = float(len(fps_window))

        if not args.quality_slider and frame_idx % 10 == 0:
            if mbps_now > args.target_mbps * 1.05 and q > q_min:
                q = max(q_min, q - 2)
            elif mbps_now < args.target_mbps * 0.80 and q < q_max:
                q = min(q_max, q + 1)

        if args.preview:
            show = frame.copy()
            cv2.putText(show, f"TX {mbps_now:.2f} Mbps | q={q} | fps~{fps_now:.0f}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(wnd, show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False

        if time.time() - last_log >= 1.0:
            print(f"[TX CUDA] bitrate={mbps_now:.2f}Mbps q={q} fps~{fps_now:.0f}")
            last_log = time.time()
        frame_idx += 1
        return True

    try:
        if not process_frame(first):
            return 0
        while True:
            tick = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            if not process_frame(frame):
                break
            if not args.max_fps:
                dt = time.time() - tick
                if dt < frame_interval:
                    time.sleep(frame_interval - dt)
    except KeyboardInterrupt:
        pass
    finally:
        api.rtc_encoder_destroy(enc)
        cap.release()
        sock.close()
        cv2.destroyAllWindows()
    return 0


def receiver_mode(args: argparse.Namespace) -> int:
    api = load_codec_api(args.codec_dll)
    dec = api.rtc_decoder_create()
    if not dec:
        print("[ERROR] rtc_decoder_create failed")
        return 1

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.bind, args.port))
    srv.listen(1)
    print(f"[INFO] listening on {args.bind}:{args.port} ...")
    conn, addr = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.setblocking(False)
    print(f"[INFO] connected: {addr[0]}:{addr[1]}")

    wnd = "live_video_chat_cuda receiver"
    cv2.namedWindow(wnd, cv2.WINDOW_NORMAL)
    last_log = time.time()
    rx_window = collections.deque()
    fps_window = collections.deque()
    raw_buf = bytearray()

    try:
        while True:
            ready_rd, _, _ = select.select([conn], [], [], 0.002)
            if ready_rd:
                while True:
                    try:
                        chunk = conn.recv(65536)
                    except BlockingIOError:
                        break
                    if not chunk:
                        raise ConnectionError("Socket closed by peer")
                    raw_buf.extend(chunk)

            while True:
                if len(raw_buf) < NET_HDR.size:
                    break
                sent_ts, payload_len = NET_HDR.unpack(raw_buf[:NET_HDR.size])
                total_len = NET_HDR.size + payload_len
                if len(raw_buf) < total_len:
                    break

                payload = bytes(raw_buf[NET_HDR.size:total_len])
                del raw_buf[:total_len]

                ts_now = time.time()
                rx_window.append((ts_now, total_len))
                fps_window.append(ts_now)
                cutoff = ts_now - 1.0
                while rx_window and rx_window[0][0] < cutoff:
                    rx_window.popleft()
                while fps_window and fps_window[0] < cutoff:
                    fps_window.popleft()

                in_arr = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
                out_ptr = ctypes.c_void_p()
                out_sz = ctypes.c_int(0)
                ow = ctypes.c_int(0)
                oh = ctypes.c_int(0)
                oc = ctypes.c_int(0)
                ok = api.rtc_decoder_decode_packet(
                    dec, in_arr, len(payload),
                    ctypes.byref(out_ptr), ctypes.byref(out_sz),
                    ctypes.byref(ow), ctypes.byref(oh), ctypes.byref(oc)
                )
                if ok == 1:
                    try:
                        rgb = ctypes.string_at(out_ptr.value, out_sz.value)
                    finally:
                        api.rtc_free_buffer(out_ptr)

                    rgb_frame = np.frombuffer(rgb, dtype=np.uint8).reshape((oh.value, ow.value, oc.value))
                    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    mbps_now = (sum(b for _, b in rx_window) * 8.0) / 1_000_000.0
                    fps_now = float(len(fps_window))
                    if args.show_latency:
                        latency_ms = (time.time() - sent_ts) * 1000.0
                        cv2.putText(
                            frame,
                            f"latency~{latency_ms:.1f} ms | RX {mbps_now:.2f} Mbps | fps~{fps_now:.0f}",
                            (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    cv2.imshow(wnd, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if time.time() - last_log >= 1.0:
                mbps = (sum(b for _, b in rx_window) * 8.0) / 1_000_000.0
                fps = float(len(fps_window))
                print(
                    f"[RX CUDA] bitrate={mbps:.2f}Mbps fps~{fps:.0f} "
                    f"buffer_bytes={len(raw_buf)}"
                )
                last_log = time.time()
    except (KeyboardInterrupt, ConnectionError):
        pass
    finally:
        api.rtc_decoder_destroy(dec)
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
    return 1


if __name__ == "__main__":
    sys.exit(main())
