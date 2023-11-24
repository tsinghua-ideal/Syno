import argparse
import json
import os, shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_latency(csv: os.PathLike):
    assert os.path.exists(csv) and os.path.splitext(csv)[1] == ".csv", os.path.splitext(
        csv
    )
    return float(pd.read_csv(csv, index_col=0).iloc[3, 0])


def identify_pareto(scores):
    # Acknowledgement: https://stackoverflow.com/questions/68284055/pareto-front-for-matplotlib-scatter-plot
    # Count number of items
    population_size = scores.shape[0]

    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)

    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return pareto_front


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

    with open(os.path.join(os.path.dirname(__file__), "baseline.json")) as f:
        baseline = json.load(f)

    assert args.model in baseline, f"{args.model} is not valid! "
    baseline_perf = baseline[args.model]

    setattr(args, "reference_acc", baseline_perf["accuracy"])
    setattr(args, "reference_flops", baseline_perf["flops"])
    setattr(args, "reference_params", baseline_perf["params"])

    if args.latency:
        args.model = args.model.split("-")[0]
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

            if args.latency and not os.path.exists(os.path.join(kernel_dir, "perf")):
                continue

            latency = (
                extract_latency(
                    os.path.join(
                        kernel_dir,
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
                    kernel_dir,
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
    os.makedirs(f"results/{args.model}-good-kernels", exist_ok=True)
    for _, kernels in all_kernels:
        for kernel in kernels:
            if kernel[1] >= args.reference_acc - 0.01:
                shutil.copytree(
                    kernel[-1],
                    f"results/{args.model}-good-kernels/{os.path.basename(kernel[-1])}",
                )

    # Accuracy vs FLOPs/param distribution

    # FLOPs
    all_flops_ratio = []
    if args.latency:
        all_latency_ratio = []
    all_y = []
    all_kernel_dir = []
    plt.figure(figsize=(10, 6), dpi=300)
    for i, (name, kernels) in enumerate(all_kernels):
        try:
            x, y, flops, params, hash_value, latency, kernel_dir = zip(
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

        all_y.extend(y)
        all_kernel_dir.extend(kernel_dir)
        flops_ratio = np.array(flops) / args.reference_flops
        all_flops_ratio.extend(list(flops_ratio))
        if args.latency:
            latency_ratio = np.array(latency) / args.reference_latency
            all_latency_ratio.extend(list(latency_ratio))

    all_flops_ratio = np.array(all_flops_ratio)
    all_y = np.array(all_y)
    plt.scatter(
        all_flops_ratio,
        all_y,
        label="FLOPs",
        s=20,
        c="#FFBE7A",
    )
    if args.latency:
        all_latency_ratio = np.array(all_latency_ratio)
        plt.scatter(
            all_latency_ratio,
            all_y,
            label="Latency",
            s=20,
            c="#82B0D2",
        )
        for f, l, acc in zip(all_flops_ratio, all_latency_ratio, all_y):
            plt.plot([f, l], [acc, acc], c="#BEB8DC", linewidth=1.0)

        # Pareto
        score = np.array([1 - all_latency_ratio, all_y]).transpose()
        pareto_mask = identify_pareto(score)

        path2perf = json.load(open("path2perf.json"))
        for msk, kernel_dir in zip(pareto_mask, all_kernel_dir):
            if not msk:
                path = json.load(open(os.path.join(kernel_dir, "meta.json")))["path"]
                accuracy = json.load(open(os.path.join(kernel_dir, "meta.json")))[
                    "accuracy"
                ]
                if accuracy > 0.695:
                    print(f"{os.path.dirname(path2perf[path])}")

        id = np.argsort(all_latency_ratio[pareto_mask])
        plt.step(
            [*all_latency_ratio[pareto_mask][id], 1.0],
            [*all_y[pareto_mask][id], args.reference_acc],
            label="Pareto",
            c="#8ECFC9",
            linewidth=1.3,
            where="post",
        )

    plt.scatter([1.0], [args.reference_acc], s=50, c="#FA7F6F", marker="^")
    plt.axhline(
        y=args.reference_acc, color="#FA7F6F", linestyle="dashed", label="acc-0"
    )
    plt.axhline(
        y=args.reference_acc - 0.01,
        color="#FA7F6F",
        linestyle="dashed",
        label="acc-0.01",
    )
    plt.axhline(
        y=args.reference_acc - 0.02,
        color="#FA7F6F",
        linestyle="dashed",
        label="acc-0.02",
    )
    plt.xlabel("FLOPs and Latency (ratio to baseline)")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.title(f"Search Result of {args.model}")
    plt.savefig(f"{args.output}-acc-vs-flops.png")

    # Params
    plt.figure(figsize=(10, 6), dpi=300)
    for i, (name, kernels) in enumerate(all_kernels):
        try:
            x, y, flops, params, hash_value, latency, kernel_dir = zip(
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
