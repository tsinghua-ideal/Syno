import argparse
import json
import os, shutil
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from plot_utils import *

def draw(ax: Axes, args):
    for model, dir, color in zip(args.models, args.dirs, ["#8ECFC9", "#FFBE7A", "#FA6F6F", "#82B0D2", "#BEB8DC"]):
        
        baseline_perf = fetch_baseline_perf(model)
        reference_acc = baseline_perf["accuracy"]
        min_acc = baseline_perf["accuracy"] - args.max_acc_decrease

        model = model.split('-')[0]
        baseline_latency = fetch_baseline_latency(model, args)

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
            linewidth=1.3,
            # where="post",
            linestyle="--",
        )
        ax.scatter(
            latency[pareto_mask][id],
            y[pareto_mask][id],
            label=model2name[model],
            s=20,
            c=color,
        )

        ax.scatter([baseline_latency], [reference_acc], s=50, c=color, marker="^")
        
        print(f"Speedup for model {model2name[model]} is from {baseline_latency / np.max(latency[pareto_mask]):.2f}x to {baseline_latency / np.min(latency[pareto_mask]):.2f}x. ")
        
    ax.set_xlabel(('GPU' if args.gpu else 'CPU') + " Inference time (ms)")

if __name__ == "__main__":
    # Get Path
    args = parser()

    assert len(args.models) == len(args.dirs)
    
    assert args.dirs is not None
    for dir in args.dirs:
        assert os.path.exists(dir) and os.path.isdir(dir)
        
    os.makedirs(args.output, exist_ok=True)
    
    fig = plt.figure(figsize=(7, 2.5), dpi=300)

    ax1 = fig.add_subplot(1, 2, 1)
    args.gpu = False
    draw(ax1, args)
    ax1.set_ylabel("Top-1 Accuracy")
    
    ax2 = fig.add_subplot(1, 2, 2)
    args.gpu = True
    draw(ax2, args)
    ax2.yaxis.set_visible(False)
    ax2.legend(loc="lower right")

    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.3, 0.5))
    
    # plt.legend(loc="lower right")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05)
    fig.savefig(os.path.join(args.output, f"imagenet-performance.pdf"))