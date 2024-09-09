bash ./run_workers.sh 8 kas KAS_r3d \
 --kas-server-addr 10.233.77.134 --kas-server-port 7070 --kas-server-save-dir results/r3d18-session-v$(date '+%Y%m%d') --kas-server-save-interval 3600\
 --kas-search-algo MCTS \
 --model torchvision/video/r3d_18 \
 --sched step --epoch 30 --batch-size 24 --warmup-epochs 0 --cooldown-epochs 0 --opt sgd --lr 1e-3 --weight-decay 1e-3 --momentum 0.9 --dataset hmdb --num-classes 51 --num-workers 16 --prune-milestones milestones/r3d_18.json --kas-inference-time-limit 600 --input-size 3 112 112 --fold 1 \
 --kas-sampler-workers 400 --kas-num-virtual-evaluator 8 \
 --kas-reward-power 4 --kas-acc-lower-bound 0.1 --kas-acc-upper-bound 0.21 \
 --kas-max-flops-ratio 1.05 --kas-min-flops-ratio 0.03 \
 --kas-max-enumerations 3 --kas-max-finalizations 2 --kas-depth 12 --kas-max-reductions 5 --kas-max-merges 3 --kas-max-splits 3 --kas-max-shifts 1 --kas-max-strides 0 --kas-max-size-multiplier 4 --kas-max-variables-in-size 4 --kas-max-chain-length 6 --kas-max-shift-rhs 2 --kas-max-expansion-merge-multiplier 2048 --kas-min-weight-share-dim 4 --kas-max-weight-share-dim 8 --kas-min-unfold-ratio 1.3 --client-mem-limit 2.3