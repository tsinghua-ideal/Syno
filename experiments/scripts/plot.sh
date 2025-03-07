python analysis/plot-models.py --dirs results/imagenet-session-reevaluate results/imagenet-session-resnet34-reevaluate results/imagenet_reevaluate_kernels/densenet121 results/imagenet_reevaluate_kernels/resnext29_2x64d results/imagenet_reevaluate_kernels/efficientnet_v2_s \
--models resnet18-imagenet resnet34-imagenet densenet121-imagenet resnext29_2x64d-imagenet efficientnet_v2_s-imagenet --output analysis/results --max-acc-decrease 0.1 --latency

python analysis/plot-models.py --dirs results/imagenet-session-reevaluate results/imagenet-session-resnet34-reevaluate --models resnet18-imagenet resnet34-imagenet --output analysis/results --max-acc-decrease 0.02 --latency

python analysis/plot.py --dirs \
 results/r3d18-session-v20240821 \
 --model r3d18-hmdb --output analysis/results/video --max-acc-decrease 0.03 --flops
 
python analysis/plot.py --dirs \
 results/r3d18-good-kernels \
 --model r3d18-hmdb --output analysis/results/video --max-acc-decrease 0.01 --flops

# python analysis/plot-models.py --dirs results/imagenet-session-reevaluate results/imagenet-session-resnet34-reevaluate \
# --models resnet18-imagenet resnet34-imagenet --output analysis/results/imagenet/resnet --max-acc-decrease 1

# python analysis/plot.py --dirs results/imagenet_reevaluate_kernels/densenet121 --model densenet121-imagenet --output analysis/results/imagenet/densenet121 --max-acc-decrease 0.05 --latency

# python analysis/plot.py --dirs results/imagenet_reevaluate_kernels/resnext29_2x64d \
# --model resnext29_2x64d-imagenet --output analysis/results/imagenet/resnext29_2x64d --max-acc-decrease 0.05 --latency

# python analysis/plot.py --dirs results/imagenet_reevaluate_kernels/efficientnet_v2_s \
# --model efficientnet_v2_s-imagenet --output analysis/results/imagenet/efficientnet_v2_s --max-acc-decrease 0.05 --latency

python analysis/plot-trends.py --dirs "results/ablation_study/MCTS_(Ours)" results/ablation_study/MCTS_without_RAVE results/ablation_study/Random_Search --output analysis/results/ablation --model resnet18

# End-to-end-performance
python analysis/end_to_end_perf_plot.py --latency --output analysis/results/

# Compression
python analysis/compression_plot.py --latency --output analysis/results/cifar100-histogram/

python analysis/layerwise.py
python analysis/plot-loss.py
python analysis/plot-quantization.py

python /cephfs/suzhengyuan/KAS-next/experiments/analysis/sync_perf.py --dirs results/resnet-good-kernels/0.2x results/resnet-good-kernels/0.4x results/resnet-good-kernels/0.5x results/resnet-good-kernels/0.6x results/resnet-good-kernels/0.7x results/densenet-good-kernels results/resnext-good-kernels results/efficientnet-good-kernels --destinations results/imagenet-session-reevaluate results/imagenet-session-resnet34-reevaluate results/imagenet_reevaluate_kernels/densenet121 results/imagenet_reevaluate_kernels/resnext29_2x64d results/imagenet_reevaluate_kernels/efficientnet_v2_s

python analysis/sync_perf_a100.py resnet-good-kernels/0.2x
python analysis/sync_perf_a100.py resnet-good-kernels/0.4x
python analysis/sync_perf_a100.py resnet-good-kernels/0.5x
python analysis/sync_perf_a100.py resnet-good-kernels/0.6x
python analysis/sync_perf_a100.py resnet-good-kernels/0.7x
python analysis/sync_perf_a100.py densenet-good-kernels
python analysis/sync_perf_a100.py resnext-good-kernels
python analysis/sync_perf_a100.py efficientnet-good-kernels
