#!/usr/bin/env python3
"""
Perception service — low-resource background daemon that:
  • Periodically grabs a webcam frame, queries the VLM and writes the
    scene description to a file an agent can read.
  • Continuously listens to the microphone and writes every recognised
    utterance to a file.
  • Optionally spawns Y-MAP-Net/scene_writer.py for structured, local
    perception (2D pose, segmentation, depth) without VLM calls.

Output directory layout (default: ./context/):
  vision_latest.txt   — latest VLM scene description
  speech_latest.txt   — latest recognised speech
  scene_latest.json   — latest YMAPNet structured observation (pose, segs, depth)
  log.jsonl           — append-only event log (one JSON object per line)
  status.json         — service heartbeat / config snapshot

Usage:
  python3 perception_service.py [options]

Examples:
  # Default (English, VLM every 30 s + YMAPNet local perception)
  python3 perception_service.py

  # Greek speech + Greek VLM answers, slower vision polling
  python3 perception_service.py --mic-lang el --greek --vision-interval 60

  # Microphone only
  python3 perception_service.py --no-vlm --no-ymapnet

  # YMAPNet structured perception only (no VLM, no mic)
  python3 perception_service.py --no-vlm --no-mic

  # All three sensors
  python3 perception_service.py --ymapnet-dir ./Y-MAP-Net
"""

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import sounddevice as sd
from gradio_client import Client, handle_file
from vosk import Model, KaldiRecognizer


# ── Utilities ────────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _fire_command(text: str, cmd_script: str = "command.sh") -> None:
    """Invoke cmd_script with *text* as a single argument (non-blocking)."""
    if not os.path.isabs(cmd_script):
        cmd_script = os.path.join(_SCRIPT_DIR, cmd_script)
    try:
        subprocess.Popen(
            [cmd_script, text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"[command] Failed to launch {cmd_script}: {e}", file=sys.stderr)


# ── Utilities ─────────────────────────────────────────────────────────────────

def ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sanitize(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[\x00-\x1F\x7F]", "", s)
    s = s.replace("\n", " ").replace("\r", "")
    return s


def write_atomic(path: str, text: str) -> None:
    """Write *text* to *path* with an atomic rename so readers never see a
    partial file."""
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


def append_log(log_path: str | None, entry: dict) -> None:
    if log_path is None:
        return
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def int_or_str(text):
    try:
        return int(text)
    except ValueError:
        return text


# ── VLM helper ────────────────────────────────────────────────────────────────

def vlm_query(client: Client, frame_path: str, prompt: str,
              temperature: float, top_p: float, max_tokens: int,
              greek: bool) -> str:
    client.predict(api_name="/reset_state")
    client.predict(
        input_images=[handle_file(frame_path)],
        input_text=prompt,
        api_name="/transfer_input",
    )
    result = client.predict(
        chatbot=[],
        temperature=temperature,
        top_p=top_p,
        max_length_tokens=max_tokens,
        repetition_penalty=1.1,
        max_context_length_tokens=4096,
        greek_translation=greek,
        api_name="/predict",
    )
    try:
        return sanitize(result[0][0][1])
    except Exception:
        return sanitize(str(result))


# ── Vision thread ─────────────────────────────────────────────────────────────

def _frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference between two frames, compared at 256×256."""
    sa = cv2.resize(a, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    sb = cv2.resize(b, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    return float(np.mean(np.abs(sa - sb)))


def vision_loop(args, out_dir: str, log_path: str, stop: threading.Event) -> None:
    from shm_camera import SHMConsumer

    consumer   = SHMConsumer(args.lib_dir, args.shm_descriptor, args.shm_stream)
    client     = None   # lazy VLM connect
    last_frame: np.ndarray | None = None  # frame used in the last successful VLM query

    with tempfile.TemporaryDirectory(prefix="perception_") as tmpdir:
        frame_path = os.path.join(tmpdir, "frame.jpg")

        while not stop.is_set():
            frame = consumer.get_frame()
            if frame is None:
                stop.wait(1)
                continue

            # ── eco: skip VLM if scene hasn't changed enough since last query ──
            if args.vision_eco > 0 and last_frame is not None:
                diff = _frame_diff(frame, last_frame)
                if diff < args.vision_eco:
                    stop.wait(args.vision_interval)
                    continue

            # Lazily connect / reconnect to VLM
            if client is None:
                try:
                    client = Client(f"http://{args.ip}:{args.port}")
                    print(f"[vision] Connected to VLM at {args.ip}:{args.port}")
                except Exception as e:
                    print(f"[vision] VLM connect failed: {e} — retry next interval",
                          file=sys.stderr)
                    stop.wait(args.vision_interval)
                    continue

            if not cv2.imwrite(frame_path, frame):
                stop.wait(args.vision_interval)
                continue

            try:
                desc = vlm_query(client, frame_path, args.prompt,
                                 args.temperature, args.top_p, args.max_tokens,
                                 args.greek)
                now = ts()
                write_atomic(
                    os.path.join(out_dir, "vision_latest.txt"),
                    f"[{now}]\n{desc}\n",
                )
                append_log(log_path, {"type": "vision", "ts": now, "description": desc})
                print(f"[vision] {now}  {desc[:100]}")
                last_frame = frame   # update reference only after a successful query
                if args.command_vlm:
                    _fire_command(f"<vision>{desc}</vision>", args.command_script)
            except Exception as e:
                print(f"[vision] Query error: {e}", file=sys.stderr)
                client = None

            stop.wait(args.vision_interval)


# ── Microphone thread ─────────────────────────────────────────────────────────

def mic_loop(args, out_dir: str, log_path: str, stop: threading.Event) -> None:
    audio_q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_, status):
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        audio_q.put(bytes(indata))

    try:
        sr = args.samplerate or int(
            sd.query_devices(args.device, "input")["default_samplerate"]
        )

        print(f"[mic] Loading Vosk model for lang={args.mic_lang} …")
        model = Model(lang=args.mic_lang)
        rec = KaldiRecognizer(model, sr)
        print(f"[mic] Listening  (sr={sr}, lang={args.mic_lang})")

        with sd.RawInputStream(
            samplerate=sr, blocksize=8000, device=args.device,
            dtype="int16", channels=1, callback=callback
        ):
            while not stop.is_set():
                try:
                    data = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                if rec.AcceptWaveform(data):
                    raw = json.loads(rec.Result())
                    text = raw.get("text", "").strip()
                    if text:
                        now = ts()
                        write_atomic(
                            os.path.join(out_dir, "speech_latest.txt"),
                            f"[{now}]\n{text}\n",
                        )
                        append_log(log_path, {"type": "speech", "ts": now, "text": text})
                        print(f"[mic]    {now}  {text}")
                        if args.command_mic and len(text.split()) > args.command_mic_words:
                            _fire_command(f"<microphone>{text}</microphone>", args.command_script)

    except Exception as e:
        print(f"[mic] Fatal: {e}", file=sys.stderr)


# ── Heartbeat thread ──────────────────────────────────────────────────────────

def heartbeat_loop(args, out_dir: str, stop: threading.Event) -> None:
    path = os.path.join(out_dir, "status.json")
    while not stop.is_set():
        payload = {
            "ts": ts(),
            "vision_interval_sec": args.vision_interval if not args.no_vlm else None,
            "mic_lang": args.mic_lang if not args.no_mic else None,
            "vlm_server": f"{args.ip}:{args.port}" if not args.no_vlm else None,
            "out_dir": os.path.abspath(out_dir),
        }
        write_atomic(path, json.dumps(payload, indent=2) + "\n")
        stop.wait(30)


# ── Command loop ─────────────────────────────────────────────────────────────

def command_loop(config_path: str, poll_interval: float, stop: threading.Event) -> None:
    """Poll configuration.json for a non-empty 'command' field.

    Fires the script named in 'command_script' with 'command' as its argument,
    once each time the value changes to something non-empty.
    """
    last_command = ""

    while not stop.is_set():
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            stop.wait(poll_interval)
            continue

        command        = cfg.get("command",        "").strip()
        command_script = cfg.get("command_script", "command.sh").strip()
        if command and command != last_command:
            last_command = command
            print(f"[command] Running: {command!r}")
            _fire_command(command, command_script)

        stop.wait(poll_interval)


# ── Camera server subprocess ──────────────────────────────────────────────────

class CameraServer:
    """Spawns camera_server.py and supervises it.

    camera_server.py is the sole process that opens the webcam; it publishes
    every frame to shared memory so that vision_loop and scene_writer can
    subscribe without contention.
    """

    def __init__(self, args):
        self.args    = args
        self._proc:  subprocess.Popen | None = None
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._supervise, daemon=True,
                                        name="camera_server")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def join(self, timeout: float = 5) -> None:
        self._thread.join(timeout=timeout)

    def _build_cmd(self) -> list[str]:
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_server.py")
        return [
            sys.executable, script,
            "--camera",     str(self.args.camera),
            "--width",      str(self.args.size[0]),
            "--height",     str(self.args.size[1]),
            "--lib-dir",    self.args.lib_dir,
            "--descriptor", self.args.shm_descriptor,
            "--stream",     self.args.shm_stream,
        ]

    def _supervise(self) -> None:
        while not self._stop.is_set():
            cmd = self._build_cmd()
            print(f"[camera] Starting camera_server")
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in self._proc.stdout:
                    if self._stop.is_set():
                        break
                    print(f"[camera] {line}", end="", flush=True)
                self._proc.wait()
                rc = self._proc.returncode
            except Exception as e:
                print(f"[camera] Launch error: {e}", file=sys.stderr)
                rc = -1

            if self._stop.is_set():
                break
            print(f"[camera] camera_server exited (rc={rc}), restarting in 3 s …",
                  file=sys.stderr)
            self._stop.wait(3)

        print("[camera] Stopped.")


# ── YMAPNet subprocess manager ───────────────────────────────────────────────

class YMAPNetSensor:
    """Spawns Y-MAP-Net/scene_writer.py as a subprocess and supervises it.

    scene_writer.py writes directly to the shared context directory, so this
    class only needs to keep the process alive and forward its stdout/stderr.
    """

    def __init__(self, ymapnet_dir: str, out_dir: str, args):
        self.ymapnet_dir = os.path.abspath(ymapnet_dir)
        self.out_dir     = os.path.abspath(out_dir)
        self.args        = args
        self._proc: subprocess.Popen | None = None
        self._stop       = threading.Event()
        self._thread     = threading.Thread(target=self._supervise, daemon=True,
                                            name="ymapnet")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def join(self, timeout: float = 5) -> None:
        self._thread.join(timeout=timeout)

    def _build_cmd(self) -> list[str]:
        # scene_writer.py is now in the project root, not inside Y-MAP-Net/
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_writer.py")
        cmd = [sys.executable, script,
               "--out-dir",      self.out_dir,
               "--ymapnet-dir",  self.ymapnet_dir,
               "--eco",          str(self.args.ymapnet_eco),
               "--interval",     str(self.args.ymapnet_interval),
               "--threshold",    str(self.args.ymapnet_threshold),
               "--vram-limit",   str(self.args.ymapnet_vram_limit),
               "--lib-dir",      self.args.lib_dir,
               "--descriptor",   self.args.shm_descriptor,
               "--stream",       self.args.shm_stream,
               ]
        if self.args.cpu:
            cmd.append("--cpu")
        if self.args.log:
            cmd.append("--log")
        return cmd

    def _supervise(self) -> None:
        """Keep scene_writer running; restart on crash until stop() is called."""
        while not self._stop.is_set():
            cmd = self._build_cmd()
            print(f"[ymapnet] Starting scene_writer: {' '.join(cmd)}")
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=self.ymapnet_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                # Stream output with prefix
                for line in self._proc.stdout:
                    if self._stop.is_set():
                        break
                    print(f"[ymapnet] {line}", end="", flush=True)
                self._proc.wait()
                rc = self._proc.returncode
            except Exception as e:
                print(f"[ymapnet] Launch error: {e}", file=sys.stderr)
                rc = -1

            if self._stop.is_set():
                break
            print(f"[ymapnet] scene_writer exited (rc={rc}), restarting in 5 s …",
                  file=sys.stderr)
            self._stop.wait(5)

        print("[ymapnet] Sensor stopped.")


# ── CLI ───────────────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configuration.json")

def load_config(path: str = CONFIG_PATH) -> dict:
    """Load configuration.json; return empty dict if missing or malformed."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"[service] Loaded config: {path}")
        return cfg
    except Exception as e:
        print(f"[service] Warning: could not read {path}: {e}", file=sys.stderr)
        return {}


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    """Build the argument parser with configuration.json values as defaults.

    Priority (highest first): CLI flag → configuration.json → built-in default.
    """
    vlm  = cfg.get("vlm",           {})
    vis  = cfg.get("vision",        {})
    mic  = cfg.get("microphone",    {})
    ym   = cfg.get("ymapnet",       {})
    out  = cfg.get("output",        {})

    shm  = cfg.get("shared_memory", {})
    cmd_hb = cfg.get("command_heartbeat", 5)

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default=CONFIG_PATH,
                   help="Path to configuration.json (default: ./configuration.json)")
    # Output
    p.add_argument("--out-dir", default=out.get("dir", "context"),
                   help="Directory for output files")
    p.add_argument("--log", action="store_true",
                   default=bool(out.get("log", False)),
                   help="Enable append-only log.jsonl (disabled by default)")
    # VLM
    p.add_argument("--ip",     default=vlm.get("ip",   "127.0.0.1"), help="VLM server host")
    p.add_argument("--port",   default=str(vlm.get("port", "8080")), help="VLM server port")
    p.add_argument("--prompt", default=vlm.get("prompt",
                   "Briefly and concisely describe what you see in this image."),
                   help="Prompt sent to the VLM")
    p.add_argument("--vision-interval", type=int,
                   default=vis.get("interval_sec", 60), metavar="SEC",
                   help="Seconds between VLM scene queries")
    p.add_argument("--vision-eco", type=float,
                   default=vis.get("eco", 0.0), metavar="THRESHOLD",
                   help="Skip VLM query when mean pixel diff vs last query is below "
                        "THRESHOLD (0 = always query, try 5–15 for static scenes)")
    p.add_argument("--no-vlm", action="store_true",
                   help="Disable VLM vision queries entirely")
    p.add_argument("--greek", action="store_true", default=bool(vlm.get("greek", False)),
                   help="Request Greek-language VLM responses")
    p.add_argument("--temperature", type=float, default=vlm.get("temperature", 0.6))
    p.add_argument("--top-p",       type=float, default=vlm.get("top_p",       0.9))
    p.add_argument("--max-tokens",  type=int,   default=vlm.get("max_tokens",  120))
    # Camera
    p.add_argument("--camera", type=int, default=vis.get("camera", 0),
                   help="Camera device index")
    # Microphone
    p.add_argument("--no-mic", action="store_true",
                   help="Disable microphone listening entirely")
    p.add_argument("--mic-lang", default=mic.get("language", "en-us"),
                   help="Vosk language code (e.g. en-us, el)")
    p.add_argument("--device", type=int_or_str, default=mic.get("device"),
                   help="Microphone device (numeric ID or name substring)")
    p.add_argument("--samplerate", type=int, default=mic.get("samplerate"),
                   help="Microphone sample rate (auto-detected if omitted)")
    # YMAPNet
    p.add_argument("--no-ymapnet", action="store_true",
                   default=not bool(ym.get("enabled", True)),
                   help="Disable YMAPNet structured perception")
    p.add_argument("--ymapnet-dir", default=ym.get("dir", "./Y-MAP-Net"),
                   help="Path to Y-MAP-Net directory")
    p.add_argument("--ymapnet-interval", type=float,
                   default=ym.get("interval_sec", 2.0), metavar="SEC",
                   help="Minimum seconds between scene_writer writes")
    p.add_argument("--ymapnet-eco", type=float, default=ym.get("eco", 5.0),
                   metavar="THRESHOLD",
                   help="Skip inference when mean pixel diff < threshold")
    p.add_argument("--ymapnet-threshold", type=float, default=ym.get("threshold", 84.0),
                   help="YMAPNet keypoint confidence threshold")
    p.add_argument("--size", nargs=2, type=int,
                   default=ym.get("size", [640, 480]),
                   metavar=("W", "H"), help="Camera resolution")
    p.add_argument("--cpu", action="store_true", default=bool(ym.get("cpu", False)),
                   help="Force CPU inference for YMAPNet")
    p.add_argument("--ymapnet-vram-limit", type=int, default=ym.get("vram_limit", 4800),
                   metavar="MB", help="GPU VRAM limit in MB for YMAPNet (default: 4800)")
    # Shared memory camera bus
    p.add_argument("--lib-dir",      default=shm.get("lib_dir", ""),
                   help="Path to SharedMemoryVideoBuffers/src/python/")
    p.add_argument("--shm-descriptor", default=shm.get("descriptor", "voithos_video.shm"))
    p.add_argument("--shm-stream",     default=shm.get("stream_name", "voithos_cam"))
    # Command watcher
    p.add_argument("--command-heartbeat", type=float, default=cmd_hb, metavar="SEC",
                   help="Seconds between configuration.json polls for a new command (default: 5)")
    p.add_argument("--command-script", default=cfg.get("command_script", "./command.sh"),
                   help="Script to invoke when relaying commands (default: ./command.sh)")
    p.add_argument("--command-mic", action="store_true",
                   default=bool(cfg.get("command_mic", False)),
                   help="Forward mic utterances to command.sh wrapped in <microphone> tags")
    p.add_argument("--command-mic-words", type=int,
                   default=cfg.get("command_mic_words", 3), metavar="N",
                   help="Minimum word count to trigger command_mic (default: 3)")
    p.add_argument("--command-vlm", action="store_true",
                   default=bool(cfg.get("command_vlm", False)),
                   help="Forward VLM descriptions to command.sh wrapped in <vision> tags")
    return p


def main() -> None:
    # First pass: pick up --config if provided, then reload with real values
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=CONFIG_PATH)
    pre_args, _ = pre.parse_known_args()

    cfg  = load_config(pre_args.config)
    args = build_parser(cfg).parse_args()

    # Check shared memory lib is configured whenever camera is needed
    need_camera = not args.no_vlm or not args.no_ymapnet
    if need_camera and not args.lib_dir:
        print("[service] ERROR: shared_memory.lib_dir not set in configuration.json",
              file=sys.stderr)
        sys.exit(1)

    # Determine whether YMAPNet scene_writer is available (now in project root)
    ymapnet_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_writer.py")
    if not args.no_ymapnet and not os.path.isfile(ymapnet_script):
        print(f"[service] scene_writer.py not found — disabling YMAPNet")
        args.no_ymapnet = True

    if args.no_vlm and args.no_mic and args.no_ymapnet:
        print("Nothing to do: all sensors disabled.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "log.jsonl") if args.log else None

    # Create context/ → out_dir symlink in the project root for convenience
    project_root = os.path.dirname(os.path.abspath(__file__))
    link_path    = os.path.join(project_root, "context")
    target       = os.path.abspath(args.out_dir)
    if not os.path.exists(link_path) and not os.path.islink(link_path):
        os.symlink(target, link_path)
        print(f"[service] Created symlink: context/ → {target}")

    print(f"[service] Perception service starting")
    print(f"[service] Output directory: {os.path.abspath(args.out_dir)}")
    if need_camera:
        print(f"[service] Camera bus: SHM '{args.shm_stream}'  ({args.shm_descriptor})")
    if not args.no_vlm:
        eco_str = f"  eco={args.vision_eco}" if args.vision_eco > 0 else ""
        print(f"[service] VLM: {args.ip}:{args.port}  interval={args.vision_interval}s{eco_str}")
    if not args.no_mic:
        print(f"[service] Mic: lang={args.mic_lang}")
    if not args.no_ymapnet:
        print(f"[service] YMAPNet: {args.ymapnet_dir}  interval={args.ymapnet_interval}s  eco={args.ymapnet_eco}")

    stop = threading.Event()
    threads:        list[threading.Thread] = []
    camera_server:  CameraServer  | None  = None
    ymapnet_sensor: YMAPNetSensor | None  = None

    def start(target, name, *extra):
        t = threading.Thread(target=target, args=(args, args.out_dir, *extra),
                             daemon=True, name=name)
        threads.append(t)
        t.start()

    # Camera server must come up before any subscriber tries to read from SHM
    if need_camera:
        camera_server = CameraServer(args)
        camera_server.start()
        print("[service] Camera server started — waiting 2 s for first frame …")
        time.sleep(2)

    if not args.no_vlm:
        start(vision_loop, "vision", log_path, stop)

    if not args.no_mic:
        start(mic_loop, "mic", log_path, stop)

    if not args.no_ymapnet:
        ymapnet_sensor = YMAPNetSensor(args.ymapnet_dir, args.out_dir, args)
        ymapnet_sensor.start()

    # Heartbeat
    hb = threading.Thread(target=heartbeat_loop, args=(args, args.out_dir, stop),
                          daemon=True, name="heartbeat")
    threads.append(hb)
    hb.start()

    # Command watcher
    cmd_t = threading.Thread(target=command_loop, args=(pre_args.config, args.command_heartbeat, stop),
                             daemon=True, name="command")
    threads.append(cmd_t)
    cmd_t.start()

    print("[service] Running — Ctrl-C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[service] Shutting down …")
        stop.set()
        if ymapnet_sensor:
            ymapnet_sensor.stop()
            ymapnet_sensor.join(timeout=5)
        if camera_server:
            camera_server.stop()
            camera_server.join(timeout=5)
        for t in threads:
            t.join(timeout=5)
        print("[service] Done.")


if __name__ == "__main__":
    main()
