# FCNet with MNIST
python train.py --model FCNet --dataset torch/mnist --mean 0.1307 --std 0.3081 --sched cosine --epoch 45 --batch-size 512 --warmup-epochs 0 --cooldown-epochs 5 --lr 0.1 --warmup-lr 0.01 --momentum 0.9 --weight-decay 0.001

# FCNet (with KAS implement) with MNIST
python train.py --model FCNet --dataset torch/mnist --mean 0.1307 --std 0.3081 --sched cosine --epoch 45 --batch-size 512 --warmup-epochs 0 --cooldown-epochs 5 --lr 0.1 --warmup-lr 0.01 --momentum 0.9 --weight-decay 0.001 --kas-replace-placeholder linear

# ConvNet with CIFAR
python train.py --model ConvNet --dataset torch/cifar10 --sched cosine --epoch 95 --batch-size 512 --warmup-epochs 0 --cooldown-epochs 5 --lr 0.1 --warmup-lr 0.01 --momentum 0.9 --weight-decay 0.001 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010

# ConvNet (with KAS implement) with CIFAR
python train.py --model ConvNet --dataset torch/cifar10 --sched cosine --epoch 95 --batch-size 512 --warmup-epochs 0 --cooldown-epochs 5 --lr 0.1 --warmup-lr 0.01 --momentum 0.9 --weight-decay 0.001 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --kas-replace-placeholder conv
