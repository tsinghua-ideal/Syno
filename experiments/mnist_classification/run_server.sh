mkdir logs
rm -rf logs/*
mkdir logs/server
python -u mcts_master.py --kas-iterations $1 > logs/server/stdout.log 2> logs/server/stderr.log