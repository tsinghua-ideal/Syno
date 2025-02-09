import matplotlib.pyplot as plt
import numpy as np
from plot_utils import *
import re

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
        accuracy = re.search(pattern, lines[-2]).group(1)

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



if __name__ == "__main__":

    args = parser()

    baseline_latency_cpu = extract_latency(os.path.join(
        args.baseline_latency_folder,
        "llvm",
        "torchvision",
        f"{model}-N=1-orig",
        "benchmark_results.csv",
    ))
    baseline_latency_gpu = extract_latency(os.path.join(
        args.baseline_latency_folder,
        "cuda",
        "torchvision",
        f"{model}-N=1-orig",
        "benchmark_results.csv",
    ))
    
    orig_cpu_performances = [
        ("Original Model", baseline_latency_cpu, 0.706787109375), 
        ("Kernel 07889", 14.3, 0.7002360224723816), 
        ("Tiled Conv", 9.72, 0.6873982548713684), 
    ]
    orig_gpu_performances = [
        ("Original Model", baseline_latency_gpu, 0.706787109375), 
        ("Kernel 07889", 3.35, 0.7002360224723816), 
        ("Tiled Conv", 4.11, 0.6873982548713684), 
    ]


    plt.figure(figsize=(6, 4), dpi=100)
    for label, latency, acc in orig_cpu_performances:
        plt.scatter(
            [latency],
            [acc],
            label=label,
            s=20,
        )
    plt.xlabel("Latency")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy and CPU Latency of Several Kernels")
    plt.tight_layout()
    plt.savefig(f"analysis/results/case-study/acc-vs-latency-cpu.png")

    plt.figure(figsize=(6, 4), dpi=100)
    for label, latency, acc in orig_gpu_performances:
        plt.scatter(
            [latency],
            [acc],
            label=label,
            s=20,
        )
    plt.xlabel("Latency")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy and GPU Latency of Several Kernels")
    plt.tight_layout()
    plt.savefig(f"analysis/results/case-study/acc-vs-latency-gpu.png")