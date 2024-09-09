python analysis/explorer.py --model torchvision/video/r3d_18 \
 --sched step --epoch 30 --batch-size 24 --warmup-epochs 0 --cooldown-epochs 0 --opt sgd --lr 1e-3 --weight-decay 1e-3 --momentum 0.9 --dataset hmdb --num-classes 51 --num-workers 10 --compile --kas-inference-time-limit 500 --input-size 3 112 112 --fold 1 \
 --kas-sampler-workers 400 --kas-num-virtual-evaluator 8 \
 --kas-reward-power 4 --kas-acc-lower-bound 0.1 --kas-acc-upper-bound 0.21 \
 --kas-max-flops-ratio 1.2 --kas-min-flops-ratio 0.15 \
 --kas-max-enumerations 3 --kas-max-finalizations 2 --kas-depth 10 --kas-max-reductions 5 --kas-max-merges 3 --kas-max-splits 3 --kas-max-shifts 1 --kas-max-strides 0 --kas-max-size-multiplier 4 --kas-max-variables-in-size 3 --kas-max-chain-length 6 --kas-max-shift-rhs 2 --kas-max-expansion-merge-multiplier 2048 --kas-min-weight-share-dim 4 --kas-max-weight-share-dim 8 --kas-min-unfold-ratio 1.3 --client-mem-limit 10.0

# visit Reduce(14869324846876338427)
# visit Reduce(14869324846876338427)
# visit Reduce(14869324846876338427)
# visit Reduce(5554949874972922426)
# visit Contraction(14297840517131538656)
# visit Unfold(13995437376865039321)
# visit Unfold(14405060147629923368)
# visit Unfold(12194851705383072312)