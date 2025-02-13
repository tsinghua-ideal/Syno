import matplotlib.pyplot as plt
import numpy as np
from plot_utils import *
import re
from matplotlib.ticker import MultipleLocator
from matplotlib.axes import Axes

orig_files = [
    "logs/imagenet/imagenet_orig.log", 
    "logs/imagenet/imagenet_ours.log", 
    "logs/imagenet/imagenet_tiledconv.log", 
]

quant_files = [
    "logs/imagenet/imagenet_orig_eval_quant.log", 
    "logs/imagenet/imagenet_ours_eval_quant.log", 
    "logs/imagenet/imagenet_tiledconv_eval_quant.log", 
]


def estimate_result(log_file, quant = False):
    with open(log_file, 'r') as f:
        lines = f.readlines()
        flops, _, _ = lines[-1][len("Evaluation result: "): ].split(" ")
        
        pattern = r"top_1=([0-9]*\.[0-9]+)"
        match = re.search(pattern, lines[-2])
        assert match
        accuracy = match.group(1)

    conv_flops = int(float(flops)) - base_flops
    if quant:
        inference_time = conv_flops / a100_int8 + base_flops / a100_tf32
    else:
        inference_time = conv_flops / a100_tf32 + base_flops / a100_tf32
        
    return inference_time, float(accuracy)

# some constant
base_flops = 10472448 # FLOPs except conv
a100_tf32 = 156 * 1e9 # 156 TFLOPS = 156 GFLOPs / ms
a100_int8 = 624 * 1e9 # 624 TFLOPS = 624 GFLOPs / ms
model = "resnet18"

def draw(ax: Axes, perf, tag): 
    xs = []
    ys = []
    for (label, latency, acc), marker in zip(perf, ['x', '^', '*', '+']):
        ax.scatter(
            [latency],
            [acc],
            label=label,
            marker=marker,
            s=50,
        )
        xs.append(latency)
        ys.append(acc)
    xs = np.array(xs[:-1])
    ys = np.array(ys[:-1])
    
    # pareto_mask = identify_pareto(np.stack([-xs, ys], axis=-1))
    # id = np.argsort(xs[pareto_mask])
    # plt.plot(xs[id], ys[id], linewidth=1.3, linestyle="--")
    ax.set_xlabel(tag + " Inference time (ms)", fontsize=14)
    ax.set_xlim(left=0)
    ax.set_ylim(0.685, 0.715)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(linestyle='--')

if __name__ == "__main__":

    args = parser()

    baseline_latency_cpu = extract_latency(os.path.join(
        args.baseline_latency_folder,
        "llvm",
        "torchvision",
        f"{model}-N=1-orig",
        "benchmark_results.csv",
    )) # type: ignore
    baseline_latency_gpu = extract_latency(os.path.join(
        args.baseline_latency_folder,
        "cuda",
        "torchvision",
        f"{model}-N=1-orig",
        "benchmark_results.csv",
    )) # type: ignore
    
    orig_cpu_performances = [
        ("Original", baseline_latency_cpu, 0.706787109375), 
        ("INT8 Quantized", 19.6, 0.6991373896598816), 
        ("Stacked Convolution", 9.72, 0.6873982548713684), 
        ("Kernel 1", 14.3, 0.7045084834098816), 
    ]
    orig_gpu_performances = [
        ("Original", baseline_latency_gpu, 0.706787109375), 
        ("INT8 Quantized", 3.25, 0.6991373896598816), # Replace GPU with new value
        ("Stacked Convolution", 4.11, 0.6873982548713684), 
        ("Kernel 1", 3.35, 0.7045084834098816), 
    ]

    fig, axs = plt.subplots(1, 2, figsize=(7, 3), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.03})
    
    ax1: Axes = axs[0] # type: ignore
    draw(ax1, orig_cpu_performances, "CPU")
    ax1.set_ylabel("Top-1 Accuracy", fontsize=14)
    
    ax2: Axes = axs[1] # type: ignore
    draw(ax2, orig_gpu_performances, "GPU")
    ax2.set_yticklabels([])

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=14, bbox_to_anchor=(0.55, 1.0))
    
    fig.subplots_adjust(top=0.73, bottom=0.2, right=0.99)
    plt.savefig(f"analysis/results/case-study.pdf")