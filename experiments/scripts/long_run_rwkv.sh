# Baseline

CUDA_VISIBLE_DEVICES=0 python test_run.py --model rwkv/rwkv-v5.1a-0.1b \
--dataset lm1b --batch-size 1 \
--lr 3e-4 --weight-decay 0.1 --grad-norm-clip 1.0 \
--gpt-seq-len 2048 --gpt-tokenizer gpt2-large --gpt-max-iters 100000 --gpt-max-minutes 0 --gpt-max-loss 100 \
--kas-max-flops-ratio 1.2 \
--kas-test-path results/rwkv-session-v20240812/04884_5463186561676786359 > logs/manual/rwkv-v2/04884_5463186561676786359.log 2>&1 &
echo $!

CUDA_VISIBLE_DEVICES=1 python test_run.py --model rwkv/rwkv-v5.1a-0.1b \
--dataset lm1b --batch-size 1 \
--lr 3e-4 --weight-decay 0.1 --grad-norm-clip 1.0 \
--gpt-seq-len 2048 --gpt-tokenizer gpt2-large --gpt-max-iters 100000 --gpt-max-minutes 0 --gpt-max-loss 100 \
--kas-max-flops-ratio 1.2 \
--kas-test-path results/rwkv-session-v20240812/04887_651368224363424942 > logs/manual/rwkv-v2/04887_651368224363424942.log 2>&1 &
echo $!

CUDA_VISIBLE_DEVICES=2 python test_run.py --model rwkv/rwkv-v5.1a-0.1b \
--dataset lm1b --batch-size 1 \
--lr 3e-4 --weight-decay 0.1 --grad-norm-clip 1.0 \
--gpt-seq-len 2048 --gpt-tokenizer gpt2-large --gpt-max-iters 100000 --gpt-max-minutes 0 --gpt-max-loss 100 \
--kas-max-flops-ratio 1.2 \
--kas-test-path results/rwkv-session-v20240812/04887_14635438364771068985 > logs/manual/rwkv-v2/04887_14635438364771068985.log 2>&1 &
echo $!

CUDA_VISIBLE_DEVICES=3 python test_run.py --model rwkv/rwkv-v5.1a-0.1b \
--dataset lm1b --batch-size 1 \
--lr 3e-4 --weight-decay 0.1 --grad-norm-clip 1.0 \
--gpt-seq-len 2048 --gpt-tokenizer gpt2-large --gpt-max-iters 100000 --gpt-max-minutes 0 --gpt-max-loss 100 \
--kas-max-flops-ratio 1.2 \
--kas-test-path results/rwkv-session-v20240812/04912_8660019745002186153 > logs/manual/rwkv-v2/04912_8660019745002186153.log 2>&1 &
echo $!

CUDA_VISIBLE_DEVICES=4 python test_run.py --model rwkv/rwkv-v5.1a-0.1b \
--dataset lm1b --batch-size 1 \
--lr 3e-4 --weight-decay 0.1 --grad-norm-clip 1.0 \
--gpt-seq-len 2048 --gpt-tokenizer gpt2-large --gpt-max-iters 100000 --gpt-max-minutes 0 --gpt-max-loss 100 \
--kas-max-flops-ratio 1.2 \
--kas-test-path results/rwkv-session-v20240812/04915_12020066856161802015 > logs/manual/rwkv-v2/04915_12020066856161802015.log 2>&1 &
echo $!

CUDA_VISIBLE_DEVICES=5 python test_run.py --model rwkv/rwkv-v5.1a-0.1b \
--dataset lm1b --batch-size 1 \
--lr 3e-4 --weight-decay 0.1 --grad-norm-clip 1.0 \
--gpt-seq-len 2048 --gpt-tokenizer gpt2-large --gpt-max-iters 100000 --gpt-max-minutes 0 --gpt-max-loss 100 \
--kas-max-flops-ratio 1.2 \
--kas-test-path results/rwkv-session-v20240812/04941_6250205952141440320 > logs/manual/rwkv-v2/04941_6250205952141440320.log 2>&1 &
echo $!

CUDA_VISIBLE_DEVICES=6 python test_run.py --model rwkv/rwkv-v5.1a-0.1b \
--dataset lm1b --batch-size 1 \
--lr 3e-4 --weight-decay 0.1 --grad-norm-clip 1.0 \
--gpt-seq-len 2048 --gpt-tokenizer gpt2-large --gpt-max-iters 100000 --gpt-max-minutes 0 --gpt-max-loss 100 \
--kas-max-flops-ratio 1.2 \
--kas-test-path results/rwkv-session-v20240812/05025_11100990547584367495 > logs/manual/rwkv-v2/05025_11100990547584367495.log 2>&1 &
echo $!

CUDA_VISIBLE_DEVICES=7 python test_run.py --model rwkv/rwkv-v5.1a-0.1b \
--dataset lm1b --batch-size 1 \
--lr 3e-4 --weight-decay 0.1 --grad-norm-clip 1.0 \
--gpt-seq-len 2048 --gpt-tokenizer gpt2-large --gpt-max-iters 100000 --gpt-max-minutes 0 --gpt-max-loss 100 \
--kas-max-flops-ratio 1.2 \
--kas-test-path results/rwkv-session-v20240812/05061_1905157613332973402 > logs/manual/rwkv-v2/05061_1905157613332973402.log 2>&1 &
echo $!

wait
