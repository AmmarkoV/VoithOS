#!/bin/bash
# claude_send.sh - Send a prompt to the running myclaude tmux session
 
SESSION="myclaude"
 
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Error: No session '$SESSION' found. Start it with: ./claude_start.sh"
  exit 1
fi
 
if [[ -z "$1" ]]; then
  echo "Usage: $0 'your prompt here'"
  echo "       echo 'your prompt' | $0"
  exit 1
fi
 
# Accept prompt from argument or stdin
if [[ "$1" == "-" ]]; then
  PROMPT=$(cat)
else
  PROMPT="$*"
fi
 
tmux send-keys -t "$SESSION" "$PROMPT" Enter
echo "Sent to '$SESSION': $PROMPT"
 
