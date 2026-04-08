#!/usr/bin/env python3
"""
talk.py — English TTS via Kokoro.

Usage:
    python3 talk.py "Hello, how are you?"
    python3 talk.py Hello world          # multiple args are joined
    echo "Hello" | python3 talk.py       # text from stdin
"""

import sys
import numpy as np
import sounddevice as sd
from kokoro import KPipeline

VOICE      = "af_bella"
LANG_CODE  = "a"          # 'a' = American English
SAMPLERATE = 24000

def speak(text: str, pipeline: KPipeline) -> None:
    for _, _, audio in pipeline(text, voice=VOICE, speed=1, split_pattern=r'\n+'):
        if isinstance(audio, (list, tuple)):
            audio = np.array(audio, dtype=np.float32)
        sd.play(audio, samplerate=SAMPLERATE)
        sd.wait()

def main() -> None:
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read()

    text = text.strip()
    if not text:
        print("talk.py: no text provided", file=sys.stderr)
        sys.exit(1)

    pipeline = KPipeline(lang_code=LANG_CODE)
    speak(text, pipeline)

if __name__ == "__main__":
    main()
