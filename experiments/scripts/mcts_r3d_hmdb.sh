bash ./run_tmux.sh 1 kas \
--kas-server-addr 0.0.0.0 --kas-server-port 7070 --kas-server-save-dir results/r3d18-session-v$(date '+%Y%m%d') --kas-server-save-interval 3600\
 --kas-search-algo MCTS \
 --model torchvision/video/r3d_18 \
 --sched cosine --epoch 20 --batch-size 20 --warmup-epochs 0 --cooldown-epochs 0 --opt adamw --lr 3e-5 --weight-decay 0.01 --dataset hmdb --num-classes 51 --num-workers 32 --compile --kas-inference-time-limit 1200 \
 --kas-sampler-workers 400 --kas-num-virtual-evaluator 4 \
 --kas-reward-power 4 --kas-acc-lower-bound 0.4 --kas-acc-upper-bound 0.7 \
 --kas-max-flops-ratio 1.2 --kas-min-flops-ratio 0.15 \
 --kas-max-enumerations 5 --kas-max-finalizations 2 --kas-depth 12 --kas-max-reductions 5 --kas-max-merges 5 --kas-max-splits 5 --kas-max-shifts 2 --kas-max-strides 2 --kas-max-size-multiplier 16 --kas-max-variables-in-size 3 --kas-max-chain-length 6 --kas-max-shift-rhs 2 --kas-max-expansion-merge-multiplier 2048 --kas-min-weight-share-dim 4 --kas-max-weight-share-dim 8 --kas-min-unfold-ratio 2.3 --client-mem-limit 1.0