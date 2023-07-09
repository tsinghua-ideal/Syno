# TODO: launch multiple clients with TMUX
python mcts_client.py --model FCNet --dataset torch/mnist --mean 0.1307 --std 0.3081 --sched cosine --epoch 5 --batch-size 512 --warmup-epochs 0 --cooldown-epochs 0 --lr 0.1 --warmup-lr 0.01 --momentum 0.9 --weight-decay 0.001 $1
