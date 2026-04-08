#!/bin/bash
# snapshot.sh — capture a webcam frame from SHM and save to long_term_context/<tag>.jpg
#
# Usage:
#   ./snapshot.sh <tag>
#   ./snapshot.sh scene        → long_term_context/scene.jpg

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ -z "$1" ]; then
    echo "Usage: snapshot.sh <tag>" >&2
    exit 1
fi

if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

exec python3 snapshot.py "$1"
