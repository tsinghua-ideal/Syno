import argparse
import json
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


if __name__ == "__main__":
    # Get Path
    parser = argparse.ArgumentParser(description="KAS session plot")
    parser.add_argument("--dirs", type=str, nargs="+", default=None)
    parser.add_argument("--output", type=str, default="plot")
    parser.add_argument("--time", default=False, action="store_true")
    parser.add_argument("--min-acc", type=float, default=0.5)
    parser.add_argument("--reference-acc", type=float, default=0.7883613782051282)
    parser.add_argument("--reference-flops", type=int, default=1823572992)
    parser.add_argument("--reference-params", type=int, default=11227812)
    args = parser.parse_args()
    assert args.dirs is not None
    for dir in args.dirs:
        assert os.path.exists(dir) and os.path.isdir(dir)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Read
    all_kernels = []
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
        if not args.time:
            kernels = list([(i, *kernels[i][1:]) for i in range(len(kernels))])
        all_kernels.append((dir, kernels))

    print(
        f"Collected {len([kernel for _, kernels in all_kernels for kernel in kernels if kernel[1] > args.min_acc])} kernels in total. "
    )
    print(
        f"The kernel with smallest FLOPs is {min([kernel for _, kernels in all_kernels for kernel in kernels if kernel[1] > args.min_acc], key=lambda x:x[2])}"
    )

    # Accuracy vs FLOPs/param distirbution

    # FLOPs
    plt.figure(figsize=(10, 6), dpi=300)
    for i, (name, kernels) in enumerate(all_kernels):
        try:
            x, y, flops, params, hash_value = zip(
                *filter(lambda x: x[1] > args.min_acc, kernels)
            )
        except ValueError:
            continue

        assert len(x) == len(y) == len(flops) == len(params) == len(hash_value)

        plt.scatter(np.array(flops) / args.reference_flops, y, label=name, s=10)
        plt.scatter([1.0], [args.reference_acc], s=50, c="r", marker="^")
    plt.xlabel("FLOPs (ratio to baseline)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{args.output}-acc-vs-flops.png")

    # Params
    plt.figure(figsize=(10, 6), dpi=300)
    for i, (name, kernels) in enumerate(all_kernels):
        try:
            x, y, flops, params, hash_value = zip(
                *filter(lambda x: x[1] > args.min_acc, kernels)
            )
        except ValueError:
            continue

        assert len(x) == len(y) == len(flops) == len(params) == len(hash_value)

        plt.scatter(np.array(params) / args.reference_params, y, label=name, s=10)
        plt.scatter([1.0], [args.reference_acc], s=50, c="r", marker="^")
    plt.xlabel("Params (ratio to baseline)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{args.output}-acc-vs-params.png")

    fig_id = 0

    # Trend figure
    fig_id += 1
    plt.figure(fig_id, figsize=(25, 6), dpi=300)
    markers = ["o", "v", "^", "<", ">", "s", "p", "*", "h", "H", "D", "d"]
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    for name, kernels in all_kernels:
        x, y, _, _, hash_value = zip(*kernels)
        y_sum, y_avg = 0, []
        for i in range(len(y)):
            y_sum += y[i]
            y_avg.append(y_sum / (i + 1))
        m, c = markers.pop(0), colors.pop(0)
        markers.append(m)
        colors.append(c)
        plt.scatter(x, y, marker=m, color=c, label=name + "_scatter", s=2)
        plt.plot(x, y_avg, marker=m, color=c, label=name, markersize=1)

    # Plot and save into file
    plt.xlabel("Time" if args.time else "Samples")
    plt.ylabel("Accuracy (avg)")
    plt.legend()
    plt.savefig(f"{args.output}-avg-acc-vs-sample.png")

    # Max figure
    fig_id += 1
    plt.figure(fig_id, figsize=(25, 6), dpi=300)
    for name, kernels in all_kernels:
        x, y, _, _, hash_value = zip(*kernels)
        y_max = []
        for i in range(len(y)):
            y_max.append(y[i] if i == 0 else max(y_max[-1], y[i]))
        y_max_log = [-math.log2(1 - y) for y in y_max]
        m, c = markers.pop(0), colors.pop(0)
        markers.append(m)
        colors.append(c)
        plt.scatter(
            x,
            [-math.log2(1 - yi) for yi in y],
            marker=m,
            color=c,
            label=name + "_scatter",
            s=2,
        )
        plt.plot(x, y_max_log, marker=m, color=c, label=name, markersize=1)

    # Plot and save into file
    plt.xlabel("Time" if args.time else "Samples")
    plt.ylabel("Accuracy (max, negative log2 scale)")
    plt.legend()
    plt.savefig(f"{args.output}-max-acc-vs-sample.png")

    # Histogram figure
    fig_id += 1
    plt.figure(fig_id, figsize=(10, 6), dpi=300)
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
