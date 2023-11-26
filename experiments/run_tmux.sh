#!/bin/bash

# Usage: ./run_tmux.sh $(NUM_GPUS) $(ENV_NAME) $(ARGS)

# Delete cache
echo 'Deleting cache...'
rm -rf .scheduler-cache
rm -rf .send-cache
rm -rf .client*-cache

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
tmux new -s "KAS" -d
tmux selectp -t 0
tmux splitw -h -p 50
for ((i = 0; i < $(log2 "$1"); i ++)); do
  for ((j = 0; j < 2 ** i; j ++)) do
    tmux selectp -t $((j * 2 + 1))
    tmux splitw -v -p 50
  done
done

# Run.
current_path=$(pwd)

logp=logs/(date +%FT%T.%3N)

mkdir -p ${logp}

# Server.
tmux send-keys -t 0 "echo TMUX Pane 1" Enter
tmux send-keys -t 0 "mamba activate $2" Enter
tmux send-keys -t 0 "cd ${current_path}" Enter
tmux send-keys -t 0 "export CUDA_VISIBLE_DEVICES=0" Enter
tmux send-keys -t 0 "./launch_server.sh ${*:3} --seed $RANDOM > ${logp}/server.log 2>&1" Enter

# Client.
for ((i = 1; i <= $1; i ++)); do
tmux send-keys -t "$i" "echo TMUX Pane $i" Enter
tmux send-keys -t "$i" "mamba activate $2" Enter
tmux send-keys -t "$i" "cd ${current_path}" Enter
tmux send-keys -t "$i" "export CUDA_VISIBLE_DEVICES=$(($i - 1))" Enter
tmux send-keys -t "$i" "sleep 5" Enter
tmux send-keys -t "$i" "./launch_client.sh ${*:3} --kas-client-cache-dir .client$(($i))-cache > ${logp}/client_$(($i)).log 2>&1" Enter
done

# Attach.
tmux attach -t "KAS"
