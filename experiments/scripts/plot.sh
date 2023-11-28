python analysis/plot-models.py --dirs results/imagenet-session-reevaluate results/imagenet-session-resnet34-reevaluate \
--model resnet18-imagenet resnet34-imagenet --output analysis/results/imagenet/resnet --max-acc-decrease 1

python analysis/plot.py --dirs results/imagenet_reevaluate_kernels/densenet121 --model densenet121-imagenet --output analysis/results/imagenet/densenet121 --max-acc-decrease 0.05 --latency

python analysis/plot.py --dirs results/imagenet_reevaluate_kernels/resnext29_2x64d \
--model resnext29_2x64d-imagenet --output analysis/results/imagenet/resnext29_2x64d --max-acc-decrease 0.05 --latency

python analysis/plot.py --dirs results/imagenet_reevaluate_kernels/efficientnet_v2_s \
--model efficientnet_v2_s-imagenet --output analysis/results/imagenet/efficientnet_v2_s --max-acc-decrease 0.05 --latency

python analysis/plot-trends.py --dirs "results/ablation_study/MCTS_(Ours)" results/ablation_study/MCTS_without_RAVE results/ablation_study/Random_Search --output analysis/results/ablation --model resnet18

# End-to-end-performance
python analysis/end_to_end_perf_plot.py --latency --output analysis/results/cifar100-histogram/max-acc-1 --max-acc-decrease 0.01

python analysis/end_to_end_perf_plot.py --latency --output analysis/cifar100-histogram/max-acc-05 --max-acc-decrease 0.005