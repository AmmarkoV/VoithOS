#!/usr/bin/env python3
"""
camera_server.py — Sole owner of the webcam.

Opens the camera once and publishes every frame to a shared memory buffer.
All other processes (vision_loop inside perception_service, scene_writer)
subscribe as read-only consumers — no camera contention.

Usage:
    python3 camera_server.py [--config configuration.json] [options]

perception_service.py spawns this automatically; you can also run it standalone.
"""

import argparse
import json
import os
import sys
import time

import cv2

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configuration.json")


def load_config(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[camera_server] Warning: could not read {path}: {e}", file=sys.stderr)
        return {}


def main() -> None:
    # Two-pass parse so --config can override the config file path
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=CONFIG_PATH)
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(pre_args.config)

    shm = cfg.get("shared_memory", {})
    vis = cfg.get("vision",        {})
    ym  = cfg.get("ymapnet",       {})

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",     default=CONFIG_PATH)
    p.add_argument("--camera",     type=int, default=vis.get("camera", 0),
                   help="Camera device index (default: 0)")
    p.add_argument("--width",      type=int, default=ym.get("size", [640, 480])[0])
    p.add_argument("--height",     type=int, default=ym.get("size", [640, 480])[1])
    p.add_argument("--lib-dir",    default=shm.get("lib_dir", ""),
                   help="Path to SharedMemoryVideoBuffers/src/python/")
    p.add_argument("--descriptor", default=shm.get("descriptor", "voithos_video.shm"),
                   help="Shared memory descriptor filename")
    p.add_argument("--stream",     default=shm.get("stream_name", "voithos_cam"),
                   help="Logical stream name")
    args = p.parse_args()

    if not args.lib_dir:
        print("[camera_server] ERROR: shared_memory.lib_dir not set in configuration.json",
              file=sys.stderr)
        sys.exit(1)

    # ── open camera ───────────────────────────────────────────────────────────
    print(f"[camera_server] Opening camera {args.camera} ({args.width}x{args.height}) …")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[camera_server] Cannot open camera {args.camera}", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Read one frame to confirm the camera works and get the actual resolution
    ret, frame = cap.read()
    if not ret:
        print("[camera_server] Cannot read first frame", file=sys.stderr)
        cap.release()
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"[camera_server] Camera opened: actual resolution {w}x{h}")

    # ── connect to shared memory ───────────────────────────────────────────────
    try:
        from shm_camera import SHMProducer
        producer = SHMProducer(args.lib_dir, args.descriptor, args.stream, w, h)
    except Exception as e:
        print(f"[camera_server] Shared memory init failed: {e}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    print(f"[camera_server] Publishing → SHM '{args.stream}'  (descriptor: {args.descriptor})")

    # Push the first frame we already captured
    producer.push(frame)

    # ── main loop ─────────────────────────────────────────────────────────────
    failed = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                failed += 1
                print(f"[camera_server] Frame read failed (#{failed})", file=sys.stderr)
                if failed > 30:
                    print("[camera_server] Too many failures — exiting", file=sys.stderr)
                    break
                time.sleep(0.1)
                continue
            failed = 0
            producer.push(frame)

    except KeyboardInterrupt:
        print("\n[camera_server] Stopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
