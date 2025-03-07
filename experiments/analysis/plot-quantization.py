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

imagenet_accuracies = {
    "Original": 0.706787109375, 
    "INT8 Quantized": 0.6991373896598816, 
    "Stacked Convolution": 0.6924235224723816, 
    "Operator 1": 0.7045084834098816,
}

def obtain_kernel_performance(folder: str, base_folder: str, scenario: str):

    if scenario == "Mobile CPU":
        split = "llvm"
    elif scenario == "Mobile GPU":
        split = "cuda"
    else:
        assert scenario == "A100"
        split = "cuda"
    
    baseline_latency = extract_latency(os.path.join(
        base_folder,
        split,
        "torchvision",
        f"{model}-N=1-orig",
        "benchmark_results.csv",
    )) # type: ignore

    quant_latency = extract_latency(os.path.join(
        base_folder,
        split,
        "qresnet18",
        "benchmark_results.csv",
    )) # type: ignore

    resnet_path = Path(folder) / "resnet-good-kernels"
    
    stacked_conv_latency = extract_latency(
        resnet_path / "ablation/Conv2d_Conv1d/perf" / split / 
        f"torchvision/{model}-N=1/benchmark_results.csv"
    )
    operator1_latency = extract_latency(
        resnet_path / "0.6x/07889_15252107013978896537/perf" / split / 
        f"torchvision/{model}-N=1/benchmark_results.csv"
    )
    operator2_latency = extract_latency(
        resnet_path / "0.2x/07754_18091915762600937904/perf" / split / 
        f"torchvision/{model}-N=1/benchmark_results.csv"
    )
    print("%s op1 speedup = %.3f x" % (scenario, baseline_latency/operator1_latency))
    print("%s op2 speedup = %.3f x" % (scenario, baseline_latency/operator2_latency))
    return [
        ("Original", baseline_latency, imagenet_accuracies["Original"]), 
        ("INT8 Quantized", quant_latency, imagenet_accuracies["INT8 Quantized"]), 
        ("Stacked Convolution", stacked_conv_latency, imagenet_accuracies["Stacked Convolution"]), 
        ("Operator 1", operator1_latency, imagenet_accuracies["Operator 1"]), 
    ]

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
    ax.set_title(tag, fontsize=14)
    ax.set_xlim(left=0)
    ax.set_ylim(0.69, 0.71)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(linestyle='--')

if __name__ == "__main__":

    args = parser()
    
    mobile_cpu_performances = obtain_kernel_performance("./results", args.baseline_latency_folder, "Mobile CPU")
    mobile_gpu_performances = obtain_kernel_performance("./results", args.baseline_latency_folder, "Mobile GPU")
    a100_performances = obtain_kernel_performance("/cephfs/shared/Syno/results", "/cephfs/shared/Syno/perf", "A100")

    fig, axs = plt.subplots(1, 3, figsize=(7, 3.5), gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.1})
    
    ax1: Axes = axs[0] # type: ignore
    draw(ax1, mobile_cpu_performances, "Mobile CPU")
    ax1.set_ylabel("Top-1 Accuracy", fontsize=14)
    ax1.set_xlim(right=45)
    ax1.xaxis.set_major_locator(MultipleLocator(10))
    
    ax2: Axes = axs[1] # type: ignore
    draw(ax2, mobile_gpu_performances, "Mobile GPU")
    ax2.set_yticklabels([])
    ax2.set_xlim(right=9)
    ax2.xaxis.set_major_locator(MultipleLocator(2))
    
    ax3: Axes = axs[2] # type: ignore
    draw(ax3, a100_performances, "A100")
    ax3.set_yticklabels([])
    ax3.set_xlim(right=1.1)
    ax3.xaxis.set_major_locator(MultipleLocator(0.3))

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=14, bbox_to_anchor=(0.55, 1.0))
    fig.supxlabel("Inference time (ms)", fontsize=14)
    
    fig.subplots_adjust(top=0.7, bottom=0.18, right=0.98)
    plt.savefig(f"analysis/results/case-study.pdf")