#!/bin/bash
# talk.sh — speak text in English using Kokoro TTS.
#
# Usage:
#   ./talk.sh "Hello, how are you?"
#   ./talk.sh Hello world
#   echo "Hello" | ./talk.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

exec python3 talk.py "$@"
