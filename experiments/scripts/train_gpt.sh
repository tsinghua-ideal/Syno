python train.py --model gpt/gpt2 \
--dataset lm1b --batch-size 1 \
--lr 3e-4 --weight-decay 0.1 --grad-norm-clip 1.0 \
--gpt-seq-len 2048 --gpt-tokenizer gpt2-large --gpt-max-iters 0 --gpt-max-minutes 10 \
--kas-replace-placeholder linear --kas-max-flops-ratio 1.2
