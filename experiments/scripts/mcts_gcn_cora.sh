./run_tmux.sh 8 torch \
--kas-server-save-dir results/gcn-session-v$(date '+%Y%m%d') \
--kas-search-algo MCTS \
--dataset Cora --lr 0.01 --weight-decay 5e-4 --epochs 200 \
--kas-sampler-workers 64 --kas-num-virtual-evaluator 8 --kas-reward-power 4 \
--kas-server-save-interval 1800 \
--kas-server-port 7070 \
--kas-max-flops-ratio 0 \
--kas-max-chain-length 6
