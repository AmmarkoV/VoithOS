#!/bin/bash
# vqa.sh — Visual Question Answering on the live camera frame.
#
# Usage:
#   ./vqa.sh "Is there anyone in the room?"
#   ./vqa.sh What colour is the wall?
#   echo "What do you see?" | ./vqa.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

exec python3 vqa.py "$@"
