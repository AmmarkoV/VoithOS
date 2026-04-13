#!/bin/bash
# claude_start.sh - Start myclaude in a named tmux session
 
SESSION="myclaude"
WORKDIR="${1:-$PWD}"  # Pass directory as arg, or use current
 
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already running. Attach with: tmux attach -t $SESSION"
  exit 0
fi
 
tmux new-session -d -s "$SESSION" -c "$WORKDIR"
tmux send-keys -t "$SESSION" \
   myclaude Enter
 
echo "Started myclaude in tmux session '$SESSION' at $WORKDIR"
echo "  Attach : tmux attach -t $SESSION"
echo "  Send   : ./claude_send.sh 'your prompt'"
 

tmux attach -t myclaude

