import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_latency(csv: os.PathLike):
    assert os.path.exists(csv) and os.path.splitext(csv) == "csv", csv
    return float(pd.read_csv(csv, index_col=0).iloc[3, 0])


if __name__ == "__main__":
    # Get Path
    parser = argparse.ArgumentParser(description="KAS session plot")
    parser.add_argument("--dirs", type=str, nargs="+", default=[])
    parser.add_argument("--output", type=str, default="plot")
    parser.add_argument("--time", default=False, action="store_true")
    parser.add_argument("--min-acc", type=float, default=0.5)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--latency", default=False, action="store_true")
    parser.add_argument(
        "--baseline-latency-folder",
        type=str,
        default="results/good_kernels/original",
    )
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(os.__file__), "baseline.json")) as f:
        baseline = json.load(f)

    assert args.model in baseline, f"{args.model} is not valid! "
    baseline_perf = baseline[args.model]

    setattr(args, "reference_acc", baseline_perf["accuracy"])
    setattr(args, "reference_flops", baseline_perf["flops"])
    setattr(args, "reference_params", baseline_perf["params"])

    if args.latency:
        baseline_file = os.path.join(
            args.baseline_latency_folder,
            "llvm",
            "torchvision",
            f"{args.model}-N=1-orig",
            "benchmark_results.csv",
        )
        baseline_latency = extract_latency(baseline_file)
        setattr(args, "reference_latency", baseline_latency)

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

            latency = (
                extract_latency(
                    os.path.join(
                        args.kernel_dir,
                        "perf",
                        "llvm",
                        "torchvision",
                        f"{args.model}-N=1",
                        "benchmark_results.csv",
                    )
                )
                if args.latency
                else 0
            )

            kernels.append(
                (
                    meta["time"],
                    meta["accuracy"],
                    meta["flops"],
                    meta["params"],
                    kernel_hash,
                    latency,
                )
            )
        kernels = sorted(kernels, key=lambda x: x[0])
        if not args.time:
            kernels = list([(i, *kernels[i][1:]) for i in range(len(kernels))])
        all_kernels.append((dir, kernels))

    print(
        f"Collected {len([kernel for _, kernels in all_kernels for kernel in kernels if kernel[1] > args.min_acc])} kernels in total."
    )
    print(
        f"The kernel with smallest FLOPs is {min([kernel for _, kernels in all_kernels for kernel in kernels if kernel[1] > args.min_acc], key=lambda x:x[2])}"
    )

    # Accuracy vs FLOPs/param distribution

    # FLOPs
    plt.figure(figsize=(10, 6), dpi=300)
    for i, (name, kernels) in enumerate(all_kernels):
        try:
            x, y, flops, params, hash_value, latency = zip(
                *filter(lambda x: x[1] > args.min_acc, kernels)
            )
        except ValueError:
            continue

        assert (
            len(x)
            == len(y)
            == len(flops)
            == len(params)
            == len(hash_value)
            == len(latency)
        )

        flops_ratio = np.array(flops) / args.reference_flops

        plt.scatter(
            flops_ratio,
            y,
            label=name + "-flops",
            s=20,
            c="y",
        )
        if args.latency:
            latency_ratio = np.array(latency) / args.reference_latency
            plt.scatter(
                latency_ratio,
                y,
                label=name + "-latency",
                s=20,
                c="b",
            )
            for f, l, acc in zip(flops_ratio, latency_ratio, y):
                plt.plot([f, l], [acc, acc], c="k")
        plt.scatter([1.0], [args.reference_acc], s=50, c="r", marker="^")
    plt.axhline(y=args.reference_acc, color="r", linestyle="dashed", label="acc-0")
    plt.axhline(
        y=args.reference_acc - 0.01, color="r", linestyle="dashed", label="acc-0.01"
    )
    plt.axhline(
        y=args.reference_acc - 0.02, color="r", linestyle="dashed", label="acc-0.02"
    )
    plt.xlabel("FLOPs (ratio to baseline)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{args.output}-acc-vs-flops.png")

    # Params
    plt.figure(figsize=(10, 6), dpi=300)
    for i, (name, kernels) in enumerate(all_kernels):
        try:
            x, y, flops, params, hash_value, latency = zip(
                *filter(lambda x: x[1] > args.min_acc, kernels)
            )
        except ValueError:
            continue

        assert (
            len(x)
            == len(y)
            == len(flops)
            == len(params)
            == len(hash_value)
            == len(latency)
        )

        plt.scatter(np.array(params) / args.reference_params, y, label=name, s=10)
        plt.scatter([1.0], [args.reference_acc], s=50, c="r", marker="^")
    plt.axhline(y=args.reference_acc, color="r", linestyle="dashed", label="acc-0")
    plt.axhline(
        y=args.reference_acc - 0.01, color="r", linestyle="dashed", label="acc-0.01"
    )
    plt.axhline(
        y=args.reference_acc - 0.02, color="r", linestyle="dashed", label="acc-0.02"
    )
    plt.xlabel("Params (ratio to baseline)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{args.output}-acc-vs-params.png")
