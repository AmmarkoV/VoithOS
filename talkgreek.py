#!/usr/bin/env python3
"""
talkgreek.py — Translate English → Greek then speak with Kokoro.

Usage:
    python3 talkgreek.py "Hello, how are you?"
    python3 talkgreek.py Hello world          # multiple args are joined
    echo "Hello" | python3 talkgreek.py       # text from stdin
    python3 talkgreek.py --greek "Γεια σου"   # skip translation, speak as-is
"""

import sys
import numpy as np
import sounddevice as sd
from kokoro import KPipeline

VOICE      = "ef_dora"
LANG_CODE  = "e"          # 'e' = Greek (espeak el)
SAMPLERATE = 24000

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

def speak(text: str, pipeline: KPipeline) -> None:
    for _, _, audio in pipeline(text, voice=VOICE, speed=1, split_pattern=r'\n+'):
        if isinstance(audio, (list, tuple)):
            audio = np.array(audio, dtype=np.float32)
        sd.play(audio, samplerate=SAMPLERATE)
        sd.wait()

def main() -> None:
    skip_translation = False
    argv = sys.argv[1:]

    if argv and argv[0] == "--greek":
        skip_translation = True
        argv = argv[1:]

    if argv:
        text = " ".join(argv)
    else:
        text = sys.stdin.read()

    text = text.strip()
    if not text:
        print("talkgreek.py: no text provided", file=sys.stderr)
        sys.exit(1)

    if not skip_translation:
        print(f"[translate] {text}", flush=True)
        text = translate_to_greek(text)
        print(f"[greek]     {text}", flush=True)

    pipeline = KPipeline(lang_code=LANG_CODE)
    speak(text, pipeline)

if __name__ == "__main__":
    main()
