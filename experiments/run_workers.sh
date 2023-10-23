#!/bin/bash

# Usage: ./run_workers.sh $(NUM_GPUS) $(ENV_NAME) $(LOG_PREFIX) $(ARGS)

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
tmux new -s "KASworker" -d
for ((i = 0; i < $(($(log2 "$1") - 1)); i ++)); do
  for ((j = 0; j < 2 ** i; j ++)) do
    tmux selectp -t $((j * 2))
    tmux splitw -v -p 50
  done
done

for ((i = 0; i <= $(($1 - 1)); i += 2)); do
    tmux selectp -t $i
    tmux splitw -h -p 50
done

# Run.
current_path=$(pwd)

# Client.
for ((i = 0; i <= $1-1; i ++)); do
tmux send-keys -t "$i" "echo TMUX Pane $i" Enter
tmux send-keys -t "$i" "mamba activate $2" Enter
tmux send-keys -t "$i" "cd ${current_path}" Enter
tmux send-keys -t "$i" "export CUDA_VISIBLE_DEVICES=$(($i))" Enter
tmux send-keys -t "$i" "./launch_client.sh ${*:4} --kas-client-cache-dir /tmp/.client_$3-$i-cache > client_$3-$i.log 2>&1" Enter
done

# Attach.
tmux attach -t "KASworker"
