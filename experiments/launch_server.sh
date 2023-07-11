rm -rf sampler-results
python server.py --model FCNet --batch-size 512 "$@"
