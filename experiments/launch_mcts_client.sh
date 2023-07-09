# TODO: launch multiple clients with TMUX
python mcts_client.py --model FCNet --dataset torch/mnist --mean 0.1307 --std 0.3081 --sched cosine --epoch 40 --batch-size 512 --warmup-epochs 0 --cooldown-epochs 10 --lr 0.3 --momentum 0.9 --weight-decay 0.001 --fetch-all-to-gpu "$@"
