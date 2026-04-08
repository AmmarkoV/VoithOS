#!/bin/bash
# talkgreek.sh — translate English → Greek then speak with Kokoro (ef_dora voice).
#
# Usage:
#   ./talkgreek.sh "Hello, how are you?"
#   ./talkgreek.sh Hello world
#   echo "Hello" | ./talkgreek.sh
#   ./talkgreek.sh --greek "Γεια σου"   # skip translation, speak Greek directly

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

exec python3 talkgreek.py "$@" 2> /dev/null
