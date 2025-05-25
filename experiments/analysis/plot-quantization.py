import matplotlib.pyplot as plt
import numpy as np
from plot_utils import parser, extract_latency
import re
import logging
from pathlib import Path
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

imagenet_accuracies = {
    "Original": 0.706787109375, 
    "INT8 Quantized": 0.7029011845588684, 
    "Stacked Convolution": 0.6924235224723816, 
    "Operator 1": 0.7045084834098816,
}

def obtain_kernel_performance(args, scenario: str):
    
    if scenario == "Mobile CPU": 
        prefix: Path = args.mdev_path
        split = "llvm"
        base_folder: Path = args.baseline_latency_folder_mdev
        stacked_conv_perf: Path = args.mdev_tiledconv_path
    elif scenario == "Mobile GPU": 
        prefix: Path = args.mdev_path
        split = "cuda"
        base_folder: Path = args.baseline_latency_folder_mdev
        stacked_conv_perf: Path = args.mdev_tiledconv_path
    else: 
        prefix: Path = args.a100_path
        split = "cuda"
        base_folder: Path = args.baseline_latency_folder_a100
        stacked_conv_perf: Path = args.a100_tiledconv_path
    
    baseline_latency = extract_latency(
        base_folder / split / f"torchvision/{model}-N=1-orig/benchmark_results.csv"
    ) # type: ignore

    quant_latency = extract_latency(
        base_folder / split / "qresnet18/benchmark_results.csv"
    ) # type: ignore
    
    stacked_conv_latency = extract_latency(
        stacked_conv_perf / "perf" / split / 
        f"torchvision/{model}-N=1/benchmark_results.csv"
    )
    operator1_latency = extract_latency(
        prefix / "resnet/07889_15252107013978896537/perf" / split / 
        f"torchvision/{model}-N=1/benchmark_results.csv"
    )
    operator2_latency = extract_latency(
        prefix / "resnet/07754_18091915762600937904/perf" / split / 
        f"torchvision/{model}-N=1/benchmark_results.csv"
    )
    logging.info("%s op1 speedup = %.3f x" % (scenario, baseline_latency/operator1_latency))
    logging.info("%s op2 speedup = %.3f x" % (scenario, baseline_latency/operator2_latency))
    return [
        ("Original", baseline_latency, imagenet_accuracies["Original"]), 
        ("INT8 Quantized", quant_latency, imagenet_accuracies["INT8 Quantized"]), 
        ("Stacked Convolution", stacked_conv_latency, imagenet_accuracies["Stacked Convolution"]), 
        ("Operator 1", operator1_latency, imagenet_accuracies["Operator 1"]), 
    ]

def draw(ax: Axes, scenario): 

    perf = obtain_kernel_performance(args, scenario)

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
    
    ax.set_title(scenario, fontsize=14)
    ax.set_xlim(left=0)
    ax.set_ylim(0.69, 0.71)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(linestyle='--')

if __name__ == "__main__":

    args = parser()

    fig, axs = plt.subplots(1, 3, figsize=(7, 3.5), gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.1})
    
    ax1: Axes = axs[0] # type: ignore
    draw(ax1, "Mobile CPU")
    ax1.set_ylabel("Top-1 Accuracy", fontsize=14)
    ax1.set_xlim(right=45)
    ax1.xaxis.set_major_locator(MultipleLocator(10))
    
    ax2: Axes = axs[1] # type: ignore
    draw(ax2, "Mobile GPU")
    ax2.set_yticklabels([])
    ax2.set_xlim(right=9)
    ax2.xaxis.set_major_locator(MultipleLocator(2))
    
    ax3: Axes = axs[2] # type: ignore
    draw(ax3, "A100")
    ax3.set_yticklabels([])
    ax3.set_xlim(right=1.1)
    ax3.xaxis.set_major_locator(MultipleLocator(0.3))

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=14, bbox_to_anchor=(0.55, 1.0))
    fig.supxlabel("Inference time (ms)", fontsize=14)
    
    fig.subplots_adjust(top=0.7, bottom=0.18, right=0.98)
    plt.savefig(args.output / "case-study.pdf")
