#!/bin/bash

# Usage: ./run_server.sh $(NUM_GPUS) $(ENV_NAME) $(ARGS)

# Check power.
if ! (( $1 > 0 && ($1 & ($1 - 1)) == 0 )); then
  echo 'Process number should be a power of 2'
  exit
fi

# Log2 function.
function log2 {
  local x=0
  for ((y = $1 - 1; y > 0; y >>= 1)); do
    ((x = x + 1))
  done
  echo "$x"
}

# Create windows.
tmux new -s "$3" -d

# Run.
current_path=$(pwd)

# Server.
tmux send-keys -t 0 "echo TMUX Pane 0" Enter
tmux send-keys -t 0 "mamba activate $2" Enter
tmux send-keys -t 0 "cd ${current_path}" Enter
tmux send-keys -t 0 "export CUDA_VISIBLE_DEVICES=0" Enter
tmux send-keys -t 0 "./launch_server.sh ${*:4} --seed $RANDOM >> server_output_$3.log 2>&1" Enter

# Attach.
tmux attach -t "$3"
