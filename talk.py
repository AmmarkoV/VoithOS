#!/usr/bin/env python3
"""
talk.py — English TTS via Kokoro.

Usage:
    python3 talk.py "Hello, how are you?"
    python3 talk.py Hello world          # multiple args are joined
    echo "Hello" | python3 talk.py       # text from stdin

    # Persistent daemon (load pipeline once, serve via FIFO):
    python3 talk.py --daemon &
    python3 talk.py "Hello"              # auto-routes to daemon if running

Options:
    -v/--voice VOICE    Kokoro voice (default: af_bella)
    --async             Return immediately, play in background
    --daemon            Run as persistent TTS daemon (keeps pipeline loaded)
"""

import argparse
import json
import os
import sys
import numpy as np
import sounddevice as sd
from kokoro import KPipeline

LANG_CODE   = "a"          # American English
SAMPLERATE  = 24000
FIFO_PATH   = "/tmp/voithos_talk.fifo"
PID_PATH    = "/tmp/voithos_talk.pid"
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configuration.json")


def _load_config() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ── playback ──────────────────────────────────────────────────────────────────

def speak(text: str, pipeline: KPipeline, voice: str, device=None) -> None:
    for _, _, audio in pipeline(text, voice=voice, speed=1, split_pattern=r'\n+'):
        if isinstance(audio, (list, tuple)):
            audio = np.array(audio, dtype=np.float32)
        sd.play(audio, samplerate=SAMPLERATE, device=device)
        sd.wait()


# ── daemon ────────────────────────────────────────────────────────────────────

def daemon_mode(voice: str) -> None:
    """Load KPipeline once, serve TTS requests arriving on FIFO."""
    if not os.path.exists(FIFO_PATH):
        os.mkfifo(FIFO_PATH)

    with open(PID_PATH, "w") as f:
        f.write(str(os.getpid()))

    print(f"[talk] Daemon PID={os.getpid()}  FIFO={FIFO_PATH}", flush=True)
    pipeline = KPipeline(lang_code=LANG_CODE)

    # Hold a write-end open so readline() on the read-end never returns EOF
    # when all external writers disconnect (classic FIFO keep-alive trick).
    write_fd = os.open(FIFO_PATH, os.O_WRONLY | os.O_NONBLOCK)
    try:
        with open(FIFO_PATH, "r") as fifo:
            while True:
                line = fifo.readline()
                if not line:
                    break
                text = line.strip()
                if text:
                    speak(text, pipeline, voice)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            os.close(write_fd)
        except OSError:
            pass
        for p in (FIFO_PATH, PID_PATH):
            try:
                os.unlink(p)
            except OSError:
                pass


# ── daemon client helpers ─────────────────────────────────────────────────────

def _daemon_alive() -> bool:
    if not os.path.exists(PID_PATH) or not os.path.exists(FIFO_PATH):
        return False
    try:
        with open(PID_PATH) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _send_to_daemon(text: str) -> bool:
    """Write one line to the daemon FIFO. Returns False if daemon is gone."""
    try:
        fd = os.open(FIFO_PATH, os.O_WRONLY | os.O_NONBLOCK)
        with os.fdopen(fd, "w") as f:
            f.write(text.replace("\n", " ") + "\n")
        return True
    except OSError:
        return False


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = _load_config()
    raw_dev = cfg.get("sound_device", None)
    try:
        default_device = int(raw_dev) if raw_dev is not None else None
    except (TypeError, ValueError):
        default_device = raw_dev

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("text", nargs="*", help="Text to speak (or read from stdin)")
    parser.add_argument("-v", "--voice", default="af_bella",
                        help="Kokoro voice (default: af_bella; "
                             "e.g. af_sarah, am_adam, bf_emma, bm_george)")
    parser.add_argument("-d", "--device", default=default_device,
                        help="Output audio device name or index (default: from configuration.json)")
    parser.add_argument("--async", dest="async_mode", action="store_true",
                        help="Return immediately while speech plays in background")
    parser.add_argument("--daemon", action="store_true",
                        help="Run as persistent TTS daemon (keeps pipeline loaded)")
    parser.add_argument("--list-devices", action="store_true",
                        help="Print available audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    if args.daemon:
        daemon_mode(args.voice)
        return

    # Collect text
    text = " ".join(args.text) if args.text else sys.stdin.read()
    text = text.strip()
    if not text:
        sys.exit(0)  # silent no-op — preferred by automation pipelines

    # Route to daemon if running
    if _daemon_alive() and _send_to_daemon(text):
        return  # daemon handles playback (inherently async)

    # Inline: --async → fork so the caller returns immediately
    if args.async_mode:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)  # parent exits right away
        os.setsid()      # detach child from terminal

    pipeline = KPipeline(lang_code=LANG_CODE)
    speak(text, pipeline, args.voice, args.device)


if __name__ == "__main__":
    main()
