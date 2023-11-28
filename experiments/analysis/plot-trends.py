import argparse
import json
import os, shutil
import math
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
import seaborn as sns
import numpy as np


if __name__ == "__main__":
    # Get Path
    parser = argparse.ArgumentParser(description="KAS session plot")
    parser.add_argument("--refreshed-dirs", type=str, nargs="+", default=[])
    parser.add_argument("--dirs", type=str, nargs="+", default=[])
    parser.add_argument("--output", type=str, default="plot")
    parser.add_argument("--summarize-dir", type=str, default="")
    parser.add_argument("--time", default=False, action="store_true")
    parser.add_argument("--min-acc", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--stop", type=int, default=200)
    args = parser.parse_args()

    assert args.dirs is not None
    for dir in args.dirs:
        assert os.path.exists(dir) and os.path.isdir(dir)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Read
    all_kernels = []
    for dir in args.refreshed_dirs:
        kernels = []
        for kernel_fmt in os.listdir(dir):
            kernel_dir = os.path.join(dir, kernel_fmt)
            if not os.path.isdir(kernel_dir):
                continue
            if "ERROR" in kernel_dir:
                continue
            if "cache" in kernel_dir:
                continue
            files = list(os.listdir(kernel_dir))
            assert "graph.dot" in files and "loop.txt" in files and "meta.json" in files

            meta_path = os.path.join(kernel_dir, "meta.json")
            meta_new_path = os.path.join(kernel_dir, "meta_new.json")
            if not os.path.exists(meta_new_path):
                continue

            with open(meta_path, "r") as f:
                meta = json.load(f)
            with open(meta_new_path, "r") as f:
                meta_new = json.load(f)

            kernel_hash = kernel_dir.split("_")[1]

            kernels.append(
                (
                    meta["time"],
                    meta_new["accuracy"],
                    meta_new["flops"],
                    meta_new["params"],
                    kernel_hash,
                )
            )
        kernels = sorted(kernels, key=lambda x: x[0])
        kernels = kernels[:args.stop]
        if not args.time:
            kernels = list([(i, *kernels[i][1:]) for i in range(len(kernels))])
        all_kernels.append((dir, kernels))

    for dir in args.dirs:
        kernels = []
        for kernel_fmt in os.listdir(dir):
            kernel_dir = os.path.join(dir, kernel_fmt)
            if not os.path.isdir(kernel_dir):
                continue
            if "ERROR" in kernel_dir:
                continue
            if "cache" in kernel_dir:
                continue
            files = list(os.listdir(kernel_dir))
            assert "graph.dot" in files and "loop.txt" in files and "meta.json" in files

            meta_path = os.path.join(kernel_dir, "meta.json")
            with open(meta_path, "r") as f:
                meta = json.load(f)

            kernel_hash = kernel_dir.split("_")[1]

            if not args.time and "time" not in meta:
                meta["time"] = 0

            kernels.append(
                (
                    meta["time"],
                    meta["accuracy"],
                    meta["flops"],
                    meta["params"],
                    kernel_hash,
                )
            )
        kernels = sorted(kernels, key=lambda x: x[0])
        kernels = kernels[:args.stop]
        if not args.time:
            kernels = list([(i, *kernels[i][1:]) for i in range(len(kernels))])
        name = ' '.join(f"{os.path.basename(dir)}".split('_'))
        all_kernels.append((name, kernels))

    # Trend figure
    plt.figure(figsize=(25, 6), dpi=300)
    for name, kernels in all_kernels:
        x, y, _, _, hash_value = zip(*kernels)
        
        ys = y
        ys = np.array(ys)
        xs = np.arange(ys.shape[0]) + 1
        y_avg = np.cumsum(ys) / xs
        y_max = np.maximum.accumulate(ys)
        y_max_log = -np.log2(0.8 - y_max)

        # plt.scatter(xs, ys, label=f"{name}-scatter", s=2)
        plt.plot(xs, y_avg, label=name, markersize=1)

    # Plot and save into file
    plt.xlabel("Time" if args.time else "Samples")
    plt.ylabel("Accuracy (avg)")
    plt.legend()
    plt.savefig(f"{args.output}-avg-acc-vs-sample.png")

    # Max figure
    plt.figure(figsize=(25, 6), dpi=300)
    for name, kernels in all_kernels:
        x, y, _, _, hash_value = zip(*kernels)
        
        ys = y
        ys = np.array(ys)
        xs = np.arange(ys.shape[0]) + 1
        y_avg = np.cumsum(ys) / xs
        y_max = np.maximum.accumulate(ys)

        # plt.scatter(
        #     xs,
        #     y_max_log,
        #     label=f"{name}-scatter",
        #     s=2,
        # )
        plt.plot(xs, y_max, label=name, markersize=1)

    # Plot and save into file
    plt.xlabel("Time" if args.time else "Samples")
    plt.ylabel("Accuracy")
    # plt.ylim(0, 0.79)
    # plt.gca().set_yscale('custom')
    plt.legend()
    plt.savefig(f"{args.output}-max-acc-vs-sample.png")

    # Histogram figure
    plt.figure(figsize=(10, 6), dpi=300)
    markers = ["o", "v", "^", "<", ">", "s", "p", "*", "h", "H", "D", "d"]
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    for name, kernels in all_kernels:
        x, y, _, _, _ = zip(*kernels)
        m, c = markers.pop(0), colors.pop(0)
        markers.append(m)
        colors.append(c)
        sns.kdeplot(y, color=c, label=name, fill=True, bw_adjust=0.2, cut=0)

    # Plot and save into file
    plt.xlabel("Accuracy")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(f"{args.output}-acc-hist.png")
