import argparse
import json
import os, shutil
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from plot_utils import *
from statistics import geometric_mean

def draw(ax: Axes, args):

    min_speedups = []
    max_speedups = []

    for model, dir, color, marker in zip(args.models, args.dirs, ["#8ECFC9", "#FFBE7A", "#FA6F6F", "#82B0D2", "#BEB8DC"], ['p', '^', '*', '+', 'x']):
        
        baseline_perf = fetch_baseline_perf(model)
        reference_acc = baseline_perf["accuracy"]
        min_acc = baseline_perf["accuracy"] - args.max_acc_decrease

        model = model.split('-')[0]
        baseline_latency = fetch_baseline_latency(model, args)
        if args.flops:
            baseline_latency = baseline_perf["flops"]

        # FLOPs
        
        kernels = []
        for kernel_fmt in os.listdir(dir):
            kernel_dir = os.path.join(dir, kernel_fmt)
            if not os.path.isdir(kernel_dir):
                continue
            if "ERROR" in kernel_dir:
                continue
            if "cache" in kernel_dir:
                continue
            if "rej" in kernel_dir:
                continue
            if "uncanon" in kernel_dir:
                continue
            files = list(os.listdir(kernel_dir))
            assert "graph.dot" in files and "loop.txt" in files and "meta.json" in files

            meta_path = os.path.join(kernel_dir, "meta.json")
            with open(meta_path, "r") as f:
                meta = json.load(f)

            kernel_hash = kernel_dir.split("_")[1]

            if not args.time and "time" not in meta:
                meta["time"] = 0

            if not os.path.exists(os.path.join(kernel_dir, "perf")):
                continue

            latency = \
                extract_latency(
                    os.path.join(
                        kernel_dir,
                        "perf",
                        "cuda" if args.gpu else "llvm",
                        "torchvision",
                        f"{model}-N=1",
                        "benchmark_results.csv",
                    )
                )

            kernels.append(
                (
                    meta["time"],
                    meta["accuracy"],
                    meta["flops"],
                    meta["params"],
                    kernel_hash,
                    latency,
                    kernel_dir,
                )
            )
        kernels = sorted(kernels, key=lambda x: x[0])
        if not args.time:
            kernels = list([(i, *kernels[i][1:]) for i in range(len(kernels))])
        
        x, y, flops, params, hash_value, latency, kernel_dir = zip(
            *filter(lambda x: x[1] > min_acc, kernels)
        )

        assert (
            len(x)
            == len(y)
            == len(flops)
            == len(params)
            == len(hash_value)
            == len(latency)
        )

        if args.flops:
            latency = flops
        
        latency = np.array(latency)
        y = np.array(y)

        # Pareto
        score = np.array([1 - latency, y]).transpose()
        pareto_mask = identify_pareto(score)

        id = np.argsort(latency[pareto_mask])
        ax.plot(
            latency[pareto_mask][id],
            y[pareto_mask][id],
            # label=f"{model}-pareto",
            c=color,
            linewidth=4,
            # where="post",
            linestyle="--",
        )
        ax.scatter(
            latency[pareto_mask][id],
            y[pareto_mask][id],
            label=model2name[model],
            s=50,
            c=color,
            marker=marker,
        )

        ax.scatter([baseline_latency], [reference_acc], s=50, c=color, marker=marker)

        min_speedup = baseline_latency / np.max(latency[pareto_mask])
        max_speedup = baseline_latency / np.min(latency[pareto_mask])
        min_speedups.append(min_speedup)
        max_speedups.append(max_speedup)
        
        print(f"Speedup for model {model2name[model]}: {min_speedup:.2f} to {max_speedup:.2f}. Ratio for model {model2name[model]}: {1 / min_speedup:.2f} to {1 / max_speedup:.2f}.  ")

    print(f"Average speedup is from {geometric_mean(min_speedups):.2f}x to {geometric_mean(max_speedups):.2f}x. ")
        
    ax.set_xlabel(('GPU' if args.gpu else 'CPU') + " Inference time (ms)", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(left=0)
    ax.grid(linestyle='--')

if __name__ == "__main__":
    # Get Path
    args = parser()

    assert len(args.models) == len(args.dirs)
    
    assert args.dirs is not None
    for dir in args.dirs:
        assert os.path.exists(dir) and os.path.isdir(dir)
        
    os.makedirs(args.output, exist_ok=True)
    
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.05})

    ax1 = axs[0]
    args.gpu = False
    draw(ax1, args)
    ax1.set_ylabel("Top-1 Accuracy", fontsize=14)
    
    ax2 = axs[1]
    args.gpu = True
    draw(ax2, args)
    ax2.set_yticklabels([])

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=14, bbox_to_anchor=(0.5, 1.0))
    
    fig.subplots_adjust(top=0.75, bottom=0.2, left=0.13, right=0.98)
    fig.savefig(os.path.join(args.output, f"imagenet-performance.pdf"))