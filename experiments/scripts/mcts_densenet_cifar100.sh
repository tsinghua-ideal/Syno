bash ./run_server.sh 8 torch KAS \
--kas-server-addr 0.0.0.0 --kas-server-port 7070 --kas-server-save-dir results/cifar100-densenet-session-v20231121 --kas-server-save-interval 3600\
 --kas-search-algo MCTS \
 --model torchvision/densenet121 \
 --kas-sampler-workers 400 --kas-num-virtual-evaluator 2 \
 --kas-reward-power 4 --kas-acc-lower-bound 0.3 --kas-acc-upper-bound 0.7 \
 --kas-max-flops-ratio 1.0 --kas-min-flops-ratio 0.15 \
 --kas-max-enumerations 5 --kas-max-finalizations 2 --kas-depth 12 --kas-max-reductions 5 --kas-max-merges 1 --kas-max-splits 3 --kas-max-shifts 2 --kas-max-strides 0 --kas-max-size-multiplier 4 --kas-max-variables-in-size 3 --kas-max-chain-length 6 --kas-max-shift-rhs 2 --kas-max-expansion-merge-multiplier 5120 --kas-min-weight-share-dim 8 --kas-max-weight-share-dim 8 --kas-min-unfold-ratio 2.3 --client-mem-limit 1.0