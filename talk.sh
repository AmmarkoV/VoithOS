#!/bin/bash
# talk.sh — translate English → Greek (via Argos) then speak with Kokoro.
#
# Usage:
#   ./talk.sh "Hello, how are you?"
#   ./talk.sh Hello world
#   echo "Hello" | ./talk.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

exec "$DIR/talkgreek.sh" "$@"
