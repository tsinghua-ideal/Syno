bash ./run_reevaluator.sh 8 torch KASreeval \
--kas-server-addr 0.0.0.0 --kas-server-port 6070 --kas-server-save-dir results/cifar100-session-reevaluate --kas-server-save-interval 3600\
 --kas-search-algo MCTS \
 --model torchvision/resnet18 \
 --kas-sampler-workers 400 --kas-num-virtual-evaluator 4 \
 --kas-reward-power 4 --kas-acc-lower-bound 0.4 --kas-acc-upper-bound 0.7 \
 --kas-max-flops-ratio 0.9 --kas-min-flops-ratio 0.15 \
 --kas-max-enumerations 5 --kas-max-finalizations 2 --kas-depth 12 --kas-max-reductions 5 --kas-max-merges 1 --kas-max-splits 3 --kas-max-shifts 2 --kas-max-strides 0 --kas-max-size-multiplier 4 --kas-max-variables-in-size 3 --kas-max-chain-length 6 --kas-max-shift-rhs 2 --kas-max-expansion-merge-multiplier 2048 --kas-min-weight-share-dim 8 --kas-max-weight-share-dim 8 --kas-min-unfold-ratio 2.3 --client-mem-limit 1.0 --kas-min-accuracy 0.768 ${*:1}