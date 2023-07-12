#!/bin/bash

# Usage: ./run_tmux.sh $(NUM_GPUS) $(ENV_NAME) $(ARGS)

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
tmux new -s "client" -d
for ((i = 0; i < $(log2 "$1")-1; i ++)); do
  for ((j = 0; j < 2 ** i; j ++)) do
    tmux selectp -t $((j * 2))
    tmux splitw -v -p 50
  done
done

for ((j = 0; j < 2 ** ($(log2 "$1")-1); j ++)) do
    tmux selectp -t $((j * 2))
    tmux splitw -h -p 50
done

# Run client.
current_path=$(pwd)
for ((i = 0; i < $1; i ++)); do
tmux send-keys -t "$i" "echo TMUX Pane $i" Enter
tmux send-keys -t "$i" "conda activate $2" Enter
tmux send-keys -t "$i" "cd ${current_path}" Enter
tmux send-keys -t "$i" "export CUDA_VISIBLE_DEVICES=$(($i))" Enter
tmux send-keys -t "$i" "bash ./launch_client.sh ${*:3}" Enter
done

# Attach.
tmux attach -t "client"