python train.py --model gcn \
--dataset Cora --lr 0.01 --weight-decay 5e-4 --epochs 200 --kas-replace-placeholder GNNLinear \
--kas-max-flops-ratio 0
