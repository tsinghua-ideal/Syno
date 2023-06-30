# FCNet with MNIST
python train.py --model FCNet --dataset torch/mnist --mean 0.1307 --std 0.3081 --sched cosine --epoch 45 --batch-size 512 --warmup-epochs 0 --cooldown-epochs 5 --lr 0.1 --warmup-lr 0.01 --momentum 0.9 --weight-decay 0.001
