#!/bin/bash
# talk.sh — translate English → Greek (via Argos) then speak with Kokoro.
#
# Usage:
#   ./talk.sh "Hello, how are you?"
#   ./talk.sh Hello world
#   echo "Hello" | ./talk.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

if [ -t 0 ]; then
    # stdin is a terminal, use command-line arguments only
    exec "$DIR/talkgreek.sh" "$@"
else
    # stdin has input, combine with command-line arguments
    stdin_input=$(cat)
    if [ -n "$stdin_input" ] && [ -n "$*" ]; then
        exec "$DIR/talkgreek.sh" "$stdin_input" "$@"
    elif [ -n "$stdin_input" ]; then
        exec "$DIR/talkgreek.sh" "$stdin_input"
    else
        exec "$DIR/talkgreek.sh" "$@"
    fi
fi
