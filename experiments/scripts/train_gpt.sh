# python train.py --model gpt/gpt2 \
# --dataset lm1b --batch-size 1 \
# --lr 3e-4 --weight-decay 0.1 --grad-norm-clip 1.0 \
# --gpt-seq-len 2048 --gpt-tokenizer gpt2-large --gpt-max-iters 0 --gpt-max-minutes 15 --gpt-max-loss 6.9 \
# --kas-max-flops-ratio 1.2

CUDA_VISIBLE_DEVICES=0 LOAD=1 python train.py --model gpt/gpt2 \
--dataset lm1b --batch-size 1 \
--lr 3e-4 --weight-decay 0.1 --grad-norm-clip 1.0 \
--gpt-seq-len 2048 --gpt-tokenizer gpt2-large --gpt-max-iters 100000 --gpt-max-minutes 0 --gpt-max-loss 10 --gpt-loss-output logs/manual/gpt2/gpt_ours_losses_6297.log \
--kas-max-flops-ratio 1.2 > logs/manual/gpt2/test_ours_6297.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 LOAD=1 python train.py --model gpt/gpt2 \
--dataset lm1b --batch-size 1 \
--lr 3e-4 --weight-decay 0.1 --grad-norm-clip 1.0 \
--gpt-seq-len 2048 --gpt-tokenizer gpt2-large --gpt-max-iters 100000 --gpt-max-minutes 0 --gpt-max-loss 10 --gpt-loss-output logs/manual/gpt2/gpt_ours_losses_nofc.log \
--kas-max-flops-ratio 1.2 > logs/manual/gpt2/test_ours_nofc.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 LOAD= python train.py --model gpt/gpt2 \
--dataset lm1b --batch-size 1 \
--lr 3e-4 --weight-decay 0.1 --grad-norm-clip 1.0 \
--gpt-seq-len 2048 --gpt-tokenizer gpt2-large --gpt-max-iters 100000 --gpt-max-minutes 0 --gpt-max-loss 10 --gpt-loss-output gpt_orig_losses.log \
--kas-max-flops-ratio 1.2 > test_orig.log 2>&1 &