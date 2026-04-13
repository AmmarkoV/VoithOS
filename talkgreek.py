#!/usr/bin/env python3
"""
talkgreek.py — Translate English → Greek then speak with Kokoro.

Usage:
    python3 talkgreek.py "Hello, how are you?"
    python3 talkgreek.py Hello world          # multiple args are joined
    echo "Hello" | python3 talkgreek.py       # text from stdin
    python3 talkgreek.py --greek "Γεια σου"   # skip translation, speak as-is
    python3 talkgreek.py --list-devices       # print available audio devices
"""

import argparse
import json
import os
import sys
import numpy as np
import sounddevice as sd
from kokoro import KPipeline

VOICE      = "ef_dora"
LANG_CODE  = "e"          # 'e' = Greek (espeak el)
SAMPLERATE = 24000
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configuration.json")


def _load_config() -> dict:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def translate_to_greek(text: str) -> str:
    from argostranslate import translate
    installed = translate.get_installed_languages()
    en = next((l for l in installed if l.code == "en"), None)
    el = next((l for l in installed if l.code == "el"), None)
    if en is None or el is None:
        print("talkgreek.py: en→el language pack not installed. "
              "Run: argospm install translate-en_el", file=sys.stderr)
        sys.exit(1)
    return en.get_translation(el).translate(text)


def speak(text: str, pipeline: KPipeline, device) -> None:
    for _, _, audio in pipeline(text, voice=VOICE, speed=1, split_pattern=r'\n+'):
        if isinstance(audio, (list, tuple)):
            audio = np.array(audio, dtype=np.float32)
        sd.play(audio, samplerate=SAMPLERATE, device=device)
        sd.wait()


def main() -> None:
    cfg = _load_config()

    # sound_device may be a name (str) or index (int)
    raw_dev = cfg.get("sound_device", None)
    try:
        default_device = int(raw_dev) if raw_dev is not None else None
    except (TypeError, ValueError):
        default_device = raw_dev  # keep as string for name-based lookup

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="*", help="Text to speak (or read from stdin)")
    parser.add_argument("--greek", action="store_true",
                        help="Skip translation; treat input as Greek text")
    parser.add_argument("-d", "--device", default=default_device,
                        help="Output audio device name or index (default: from configuration.json)")
    parser.add_argument("--list-devices", action="store_true",
                        help="Print available audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        sys.exit(0)

    text = " ".join(args.text) if args.text else sys.stdin.read()
    text = text.strip()
    if not text:
        sys.exit(0)

    if not args.greek:
        print(f"[translate] {text}", flush=True)
        text = translate_to_greek(text)
        print(f"[greek]     {text}", flush=True)

    pipeline = KPipeline(lang_code=LANG_CODE)
    speak(text, pipeline, args.device)


if __name__ == "__main__":
    main()
