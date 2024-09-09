import copy
from matplotlib import rcParams
import argparse
import os, shutil, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from tqdm import tqdm

ansor_color = '#a7d0e3'
nas_pte_color = '#fa8074'
micro_nas_color = '#d56050'
micro_nas_compress_color = '#f0787a'
seq1_color = '#c2deeb'
seq2_color = '#90c4dc'
seq3_color = nas_pte_color

model2name = {
    "resnet18": "ResNet18", 
    "resnet34": "ResNet34", 
    "densenet121": "DenseNet121", 
    "efficientnet_v2_s": "EfficientNet-V2-S", 
    "resnext29_2x64d": "ResNeXt29-2x64d", 
}


def get_data_dims(data: list, dim):
    return [item[dim] for item in data]


def check_format(entries: []):
    assert isinstance(entries, list)
    assert len(entries) > 0

    for entry in entries:
        assert isinstance(entry, dict)
        assert 'name' in entry and isinstance(entry['name'], str)
        assert 'data' in entry and isinstance(entry['data'], list)
        assert len(entry['data']) > 0
        if 'baseline' in entry:
            assert isinstance(entry['baseline'], bool)
        if 'text_mark' in entry:
            assert isinstance(entry['text_mark'], bool)
        for item in entry['data']:
            assert isinstance(item, tuple) and len(item) == 2
            key, value = item
            assert isinstance(key, str)
            assert isinstance(value, float) or isinstance(value, int)

    first_data_keys = get_data_dims(entries[0]['data'], 0)
    for entry in entries[1:]:
        entry_data_keys = get_data_dims(entry['data'], 0)
        assert first_data_keys == entry_data_keys, \
            f'Data entry mismatch: {first_data_keys} v.s. {entry_data_keys}'


def check_baseline(entries: [], relative: bool):
    count, baseline = 0, None
    for entry in entries:
        if 'baseline' in entry and entry['baseline']:
            count += 1
            baseline = entry
    assert count <= 1, f'Should be less than 1 baseline, {count} found'
    if count == 0:
        assert not relative
    return copy.deepcopy(baseline) if baseline and relative else None


def simplify(entries: [], baseline: dict, speedup: bool):
    names = get_data_dims(entries, 'name')
    labels = get_data_dims(entries[0]['data'], 0)
    bars = []
    for i in range(len(labels)):
        comparisons = []
        for entry in entries:
            value = entry['data'][i][1]
            if baseline is not None:
                value /= baseline['data'][i][1]
                value = 1 / value if speedup else value
            comparisons.append(value)
        bars.append(comparisons)
    return names, labels, bars


def text_numbers(ax, width, entries, bars,
                 fontsize=rcParams['font.size'], fontweight=None,
                 extra_height: float = 0.03):
    num_entries = len(entries)
    width_per_entry = width / num_entries
    num_groups = len(entries[0]['data'])

    for i, entry in enumerate(entries):
        if 'text_mark' in entry and entry['text_mark']:
            for j in range(num_groups):
                value = bars[j][i]
                ax.text(j - width / 2 + width_per_entry * (i + 0.5), value + extra_height,
                        '{:.2f}×'.format(value), fontsize=fontsize, fontweight=fontweight,
                        ha='center')


def text_numbers_custom(ax, width, entries, bars, 
                 fontsize=rcParams['font.size'], fontweight=None):
    num_entries = len(entries)
    width_per_entry = width / num_entries
    num_groups = len(entries[0]['data'])

    for i, entry in enumerate(entries):
        if 'text_mark' in entry and entry['text_mark']:
            assert len(entry['offset']) == num_groups
            for j, (xo, yo) in enumerate(entry['offset']):
                value = bars[j][i]
                ax.text(j - width / 2 + width_per_entry * (i + 0.5) + xo, value + yo,
                        '{:.2f}×'.format(value), fontsize=fontsize, fontweight=fontweight,
                        ha='center')

def extract_latency(csv: os.PathLike):
    assert os.path.exists(csv) and os.path.splitext(csv)[1] == ".csv", os.path.splitext(
        csv
    )
    return float(pd.read_csv(csv, index_col=0).iloc[3, 0]) * 1000

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

def fetch_baseline_perf(model):
    with open(os.path.join(os.path.dirname(__file__), "baseline.json")) as f:
        baseline = json.load(f)
    assert model in baseline, f"{model} is not valid! "
    baseline_perf = baseline[model]

    return baseline_perf

def fetch_baseline_latency(model, args):
    baseline_file = os.path.join(
        args.baseline_latency_folder,
        "llvm",
        "torchvision",
        f"{model}-N=1-orig",
        "benchmark_results.csv",
    )
    baseline_latency = extract_latency(baseline_file)
    return baseline_latency

def parser():
    parser = argparse.ArgumentParser(description="KAS session plot")
    parser.add_argument("--dirs", type=str, nargs="+", default=[])
    parser.add_argument("--output", type=str, default="analysis/results")
    parser.add_argument("--time", default=False, action="store_true")
    parser.add_argument("--max-acc-decrease", type=float, default=0.01)
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--models", type=str, nargs="+", default=[])
    parser.add_argument("--flops", default=False, action="store_true")
    parser.add_argument("--latency", default=False, action="store_true")
    parser.add_argument("--offset-bar", default=False, action="store_true")
    parser.add_argument(
        "--baseline-latency-folder",
        type=str,
        default="results/good_kernels/original",
    )
    args = parser.parse_args()
    
    baseline_perf = fetch_baseline_perf(args.model)

    setattr(args, "reference_flops", baseline_perf["flops"])
    setattr(args, "reference_params", baseline_perf["params"])
    if "gpt" in args.model:
        setattr(args, "reference_loss", baseline_perf["loss"])
        setattr(args, "max_loss", baseline_perf["loss"] + args.max_acc_decrease)
    else:
        setattr(args, "reference_acc", baseline_perf["accuracy"])
        setattr(args, "min_acc", baseline_perf["accuracy"] - args.max_acc_decrease)
    
    if args.latency:
        baseline_latency = fetch_baseline_latency(args.model.split("-")[0], args)
        setattr(args, "reference_latency", baseline_latency)
    
    assert args.dirs is not None
    for dir in args.dirs:
        assert os.path.exists(dir) and os.path.isdir(dir)

    os.makedirs(args.output, exist_ok=True)
    
    return args

def collect_kernels(dir, model, args):
    subdirs = os.listdir(dir)
    
    if any(re.match("0\.\dx", subdir) for subdir in subdirs):
        results = [collect_kernels(os.path.join(dir, subdir), model, args) for subdir in subdirs if re.match("0\.\dx", subdir)]
        return [k for res in results for k in res]
    
    kernels = []
    for kernel_fmt in tqdm(subdirs):
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
        if os.path.exists(os.path.join(kernel_dir, "meta_new.json")):
            meta_path = os.path.join(kernel_dir, "meta_new.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        if meta["accuracy"] < args.min_acc:
            continue

        kernel_hash = kernel_dir.split("_")[1]

        if not args.time and "time" not in meta:
            meta["time"] = 0

        if args.latency and not os.path.exists(os.path.join(
                    kernel_dir,
                    "perf",
                    "llvm",
                    "torchvision",
                    f"{model}-N=1",
                    "benchmark_results.csv",
                )):
            continue

        latency = (
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
            if args.latency
            else 0
        )
        
        if "loss" not in meta:
            meta["loss"] = 0

        kernels.append(
            (
                meta["time"],
                meta["accuracy"],
                meta["flops"],
                meta["params"],
                meta["loss"],
                kernel_hash,
                latency,
                kernel_dir,
            )
        )
    kernels = sorted(kernels, key=lambda x: x[0])
    if not args.time:
        kernels = list([(i, *kernels[i][1:]) for i in range(len(kernels))])

    return kernels

def plot_baseline(model, args):
    if "gpt" in model:
        plt.scatter([1.0], [args.reference_loss], s=50, c="#FA7F6F", marker="^")
        if args.offset_bar:
            plt.axhline(
                y=args.reference_loss, color="#FA7F6F", linestyle="dashed", label="baseline loss"
            )
            plt.axhline(
                y=args.max_loss,
                color="#FA7F6F",
                linestyle="dashed",
                label="Max loss",
            )
    else:
        plt.scatter([1.0], [args.reference_acc], s=50, c="#FA7F6F", marker="^")
        if args.offset_bar:
            plt.axhline(
                y=args.reference_acc, color="#FA7F6F", linestyle="dashed", label="baseline accuracy"
            )
            plt.axhline(
                y=args.min_acc,
                color="#FA7F6F",
                linestyle="dashed",
                label="Min accuracy",
            )