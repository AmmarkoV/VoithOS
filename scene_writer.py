#!/usr/bin/env python3
"""
scene_writer.py — YMAPNet structured perception subscriber.

Reads frames from the shared memory camera bus (published by camera_server.py),
runs YMAPNet inference in fast/headless mode, and writes structured scene
observations to the shared context directory.

Must be run from VoithOS/ (project root), not from inside Y-MAP-Net/.

Output files:
    <out-dir>/scene_latest.json   — latest structured observation
    <out-dir>/log.jsonl           — append-only event log (type=scene entries)

JSON schema per observation:
    {
      "type": "scene",
      "ts": "2026-04-09T14:32:10",
      "frame": 42,
      "description": "person sitting at desk with laptop",
      "keypoints": { "nose": [320, 240, 198.4], ... },
      "num_keypoints": 9,
      "scores": {
        "person": 0.82, "face": 0.61, "hand": 0.42, "foot": 0.12,
        "text": 0.05, "vehicle": 0.00, "animal": 0.00,
        "object": 0.35, "furniture": 0.71
      },
      "depth_mean": 142.3
    }

Usage (spawned automatically by perception_service.py):
    python3 scene_writer.py --ymapnet-dir ./Y-MAP-Net [options]
"""

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime

import cv2
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configuration.json")


# ── config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[scene_writer] Warning: cannot read config {path}: {e}", file=sys.stderr)
        return {}


# ── helpers ───────────────────────────────────────────────────────────────────

def ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_atomic(path: str, text: str) -> None:
    dir_ = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=dir_)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def append_log(log_path: str, entry: dict) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def heatmap_score(hm: np.ndarray) -> float:
    """Activation score [0, 1] for a uint8 heatmap (baseline ≈ 120)."""
    return max(0.0, min(1.0, (float(np.max(hm)) - 120.0) / 135.0))


def build_scene_snapshot(estimator, frame_number: int) -> dict:
    kp_names = estimator.keypoint_names
    kp_res   = getattr(estimator, "keypoint_results", [])

    keypoints = {}
    for i, name in enumerate(kp_names):
        if i < len(kp_res) and kp_res[i]:
            x, y, conf = kp_res[i][0]
            keypoints[name] = [round(float(x), 1), round(float(y), 1), round(float(conf), 2)]

    hm = estimator.heatmapsOut

    def score(ch: int) -> float:
        if ch < 0 or ch >= len(hm):
            return 0.0
        return round(heatmap_score(hm[ch]), 3)

    scores = {
        "person":    score(39),
        "face":      score(40),
        "hand":      score(41),
        "foot":      score(42),
        "text":      score(estimator.chanText),
        "vehicle":   score(estimator.chanVehicle),
        "animal":    score(estimator.chanAnimal),
        "object":    score(estimator.chanObject),
        "furniture": score(estimator.chanFurniture),
    }

    depth_mean = 0.0
    if estimator.depthmap is not None:
        depth_mean = round(float(np.mean(estimator.depthmap)), 2)

    return {
        "type":          "scene",
        "ts":            ts(),
        "frame":         frame_number,
        "description":   (estimator.description or "").strip(),
        "keypoints":     keypoints,
        "num_keypoints": len(keypoints),
        "scores":        scores,
        "depth_mean":    depth_mean,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=CONFIG_PATH)
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(pre_args.config)

    shm = cfg.get("shared_memory", {})
    ym  = cfg.get("ymapnet",       {})
    out = cfg.get("output",        {})

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",      default=CONFIG_PATH)
    p.add_argument("--out-dir",     default=out.get("dir", "./context"))
    p.add_argument("--ymapnet-dir", default=ym.get("dir",  "./Y-MAP-Net"),
                   help="Path to Y-MAP-Net directory")
    p.add_argument("--model",       default="2d_pose_estimation")
    p.add_argument("--engine",      default="tensorflow")
    p.add_argument("--threshold",   type=float, default=ym.get("threshold", 84.0))
    p.add_argument("--interval",    type=float, default=ym.get("interval_sec", 2.0),
                   help="Minimum seconds between writes")
    p.add_argument("--eco",         type=float, default=ym.get("eco", 5.0),
                   help="Skip inference when mean pixel diff < ECO")
    p.add_argument("--cpu",         action="store_true", default=bool(ym.get("cpu", False)))
    # Shared memory
    p.add_argument("--lib-dir",     default=shm.get("lib_dir", ""),
                   help="Path to SharedMemoryVideoBuffers/src/python/")
    p.add_argument("--descriptor",  default=shm.get("descriptor", "voithos_video.shm"))
    p.add_argument("--stream",      default=shm.get("stream_name", "voithos_cam"))
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # ── validate paths ────────────────────────────────────────────────────────
    ymapnet_dir = os.path.abspath(args.ymapnet_dir)
    if not os.path.isdir(ymapnet_dir):
        print(f"[scene_writer] Y-MAP-Net directory not found: {ymapnet_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.lib_dir:
        print("[scene_writer] ERROR: shared_memory.lib_dir not set in configuration.json",
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path   = os.path.join(args.out_dir, "log.jsonl")
    scene_path = os.path.join(args.out_dir, "scene_latest.json")

    # ── connect to shared memory camera ──────────────────────────────────────
    from shm_camera import SHMConsumer
    consumer = SHMConsumer(args.lib_dir, args.descriptor, args.stream)
    print(f"[scene_writer] Waiting for camera stream '{args.stream}' …")

    # Wait until first frame arrives before loading YMAPNet (saves startup time)
    first_frame = None
    while first_frame is None:
        first_frame = consumer.get_frame()
        if first_frame is None:
            time.sleep(0.5)

    print(f"[scene_writer] Camera stream ready: {first_frame.shape[1]}x{first_frame.shape[0]}")

    # ── import YMAPNet from its directory ─────────────────────────────────────
    if ymapnet_dir not in sys.path:
        sys.path.insert(0, ymapnet_dir)

    # YMAPNet loads model files relative to cwd — run from its directory
    original_dir = os.getcwd()
    os.chdir(ymapnet_dir)

    try:
        print("[scene_writer] Loading YMAPNet model …")
        from YMAPNet import YMAPNet  # noqa: PLC0415

        estimator = YMAPNet(
            modelPath=args.model,
            threshold=int(args.threshold),
            keypoint_threshold=args.threshold,
            engine=args.engine,
            profiling=False,
            illustrate=False,
            pruneTokens=False,
            monitor=[],
            window_arrangement=[],
            screen_w=1920,
            screen_h=1080,
            depth_iterations=0,        # fast mode
            estimate_person_id=False,
            resolve_skeleton=False,
        )
    finally:
        os.chdir(original_dir)

    print(f"[scene_writer] Ready — writing to {os.path.abspath(args.out_dir)}")

    last_write   = 0.0
    prev_input: np.ndarray | None = None

    try:
        while True:
            frame = consumer.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            # ── eco: skip inference when scene is static ──────────────────
            small = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
            if args.eco > 0 and prev_input is not None:
                diff = float(np.mean(np.abs(small.astype(np.float32) - prev_input)))
                if diff < args.eco:
                    continue
            prev_input = small.astype(np.float32)

            # ── run inference (YMAPNet expects cwd = its own directory) ────
            os.chdir(ymapnet_dir)
            try:
                estimator.process(frame, static_frame_threshold=0.0)
            except Exception as e:
                print(f"[scene_writer] Inference error: {e}", file=sys.stderr)
                continue
            finally:
                os.chdir(original_dir)

            # ── rate-limit writes ─────────────────────────────────────────
            now = time.time()
            if args.interval > 0 and (now - last_write) < args.interval:
                continue
            last_write = now

            # ── write observation ─────────────────────────────────────────
            try:
                snap = build_scene_snapshot(estimator, estimator.frameNumber)
            except Exception as e:
                print(f"[scene_writer] Snapshot error: {e}", file=sys.stderr)
                continue

            write_atomic(scene_path, json.dumps(snap, indent=2, ensure_ascii=False) + "\n")
            append_log(log_path, snap)
            print(f"[scene] {snap['ts']}  kp={snap['num_keypoints']:2d}  "
                  f"person={snap['scores']['person']:.2f}  {snap['description'][:60]}",
                  flush=True)

    except KeyboardInterrupt:
        print("\n[scene_writer] Stopped.")


if __name__ == "__main__":
    main()
