rm -rf sampler-results
python mcts_server.py --model FCNet --batch-size 512 "$@"
