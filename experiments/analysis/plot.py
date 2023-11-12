import argparse
import json
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == "__main__":
    # Get Path
    parser = argparse.ArgumentParser(description="KAS session plot")
    parser.add_argument("--dirs", type=str, nargs="+", default=None)
    parser.add_argument("--output", type=str, default="plot")
    parser.add_argument("--time", default=False, action="store_true")
    parser.add_argument("--min-acc", type=float, default=0.5)
    parser.add_argument("--reference-hash", type=str, default=None)
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

    fig_id = 0

    # Accuracy vs FLOPs/param distirbution
    for i, (name, kernels) in enumerate(all_kernels):
        x, y, flops, params, hash_value = zip(
            *filter(lambda x: x[1] > args.min_acc, kernels)
        )

        assert len(x) == len(y) == len(flops) == len(params) == len(hash_value)
        if args.reference_hash is not None:
            try:
                ind = hash_value.index(args.reference_hash)
            except ValueError as e:
                print(f"{args.reference_hash} does not exists in {name}, {hash_value}")
                exit(1)

        # FLOPs
        fig_id += 1
        plt.figure(fig_id, figsize=(10, 6), dpi=300)
        plt.scatter(flops, y, label=name, s=3)
        if args.reference_hash is not None:
            plt.scatter([flops[ind]], [y[ind]], s=5, c="r", marker="^")
        plt.xlabel("FLOPs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"{args.output}-acc-vs-flops-{i}.png")

        # Params
        fig_id += 1
        plt.figure(fig_id, figsize=(10, 6), dpi=300)
        plt.scatter(params, y, label=name, s=3)
        if args.reference_hash is not None:
            plt.scatter([params[ind]], [y[ind]], s=5, c="r", marker="^")
        plt.xlabel("Params")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"{args.output}-acc-vs-params-{i}.png")

    # Trend figure
    fig_id += 1
    plt.figure(fig_id, figsize=(25, 6), dpi=300)
    markers = ["o", "v", "^", "<", ">", "s", "p", "*", "h", "H", "D", "d"]
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    for name, kernels in all_kernels:
        x, y, _, _, hash_value = zip(
            *filter(lambda k: k[-1] != args.reference_hash, kernels)
        )
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
        x, y, _, _, hash_value = zip(
            *filter(lambda k: k[-1] != args.reference_hash, kernels)
        )
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
