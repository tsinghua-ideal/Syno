python train.py --model gpt/gpt-nano \
--dataset lm1b --batch-size 64 \
--lr 3e-4 --weight-decay 0.1 --grad-norm-clip 1.0 \
--gpt-seq-len 1024 --gpt-tokenizer gpt2-large --gpt-max-iters 0
