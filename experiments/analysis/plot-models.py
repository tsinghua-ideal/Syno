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
    parser.add_argument("--max-acc-decrease", type=float, default=0.02)
    parser.add_argument("--model", type=str, nargs="+", default=["resnet18", "resnet34"])
    parser.add_argument(
        "--baseline-latency-folder",
        type=str,
        default="results/good_kernels/original",
    )
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), "baseline.json")) as f:
        baseline = json.load(f)

    assert len(args.model) == len(args.dirs)
    assert all(model in baseline for model in args.model), f"{args.model} is not valid! "   
    
    assert args.dirs is not None
    for dir in args.dirs:
        assert os.path.exists(dir) and os.path.isdir(dir)
        
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    plt.figure(figsize=(10, 6), dpi=300)
    for model, dir, color in zip(args.model, args.dirs, ["#82B0D2", "#8ECFC9"]):
        baseline_perf = baseline[model]
        reference_acc = baseline_perf["accuracy"]
        min_acc = baseline_perf["accuracy"] - args.max_acc_decrease

        model = model.split("-")[0]
        baseline_file = os.path.join(
            args.baseline_latency_folder,
            "llvm",
            "torchvision",
            f"{model}-N=1-orig",
            "benchmark_results.csv",
        )
        baseline_latency = extract_latency(baseline_file)

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
                        "llvm",
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
        print(f"collected  {len(kernels)} kernels for {model}")
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
        
        plt.scatter(
            latency,
            y,
            label=model,
            s=20,
            c=color,
        )

        # Pareto
        score = np.array([1 - latency, y]).transpose()
        pareto_mask = identify_pareto(score)

        id = np.argsort(latency[pareto_mask])
        plt.plot(
            latency[pareto_mask][id],
            y[pareto_mask][id],
            label="Pareto",
            c="#8ECFC9",
            linewidth=1.3,
            # where="post",
            linestyle="--",
        )

        plt.scatter([baseline_latency], [reference_acc], s=50, c=color, marker="^", label=f"{model}-baseline")
        
    plt.xlabel("Latency (ms)")
    plt.ylabel("ImageNet Classification Accuracy")
    plt.legend(loc="lower right")
    plt.title(f"Search Result")
    plt.savefig(f"{args.output}-acc-vs-latency.png")