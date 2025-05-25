import os, logging
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator
import numpy as np
from plot_utils import parser, collect_kernels, fetch_baseline_latency, fetch_baseline_perf, model2name, identify_pareto, Scenarios
from statistics import geometric_mean

def draw(ax: Axes, args, scenario: str, use_inductor: bool=False):
    
    if scenario in ["Mobile CPU", "Mobile GPU"]: 
        prefix = args.mdev_path
        args.gpu = (scenario == "Mobile GPU")
        baseline_latency_folder = args.baseline_latency_folder_mdev
    else: 
        prefix = args.a100_path
        args.gpu = True
        baseline_latency_folder = args.baseline_latency_folder_a100

    min_speedups = []
    max_speedups = []

    for model, dir, color, marker in zip(args.models, args.dirs, ["#8ECFC9", "#FFBE7A", "#FA6F6F", "#82B0D2", "#BEB8DC"], ['p', '^', '*', 'P', 'd']):
        
        baseline_perf = fetch_baseline_perf(model)
        reference_acc = baseline_perf["accuracy"]
        min_acc = baseline_perf["accuracy"] - args.max_acc_decrease
        args.min_acc = min_acc

        model = model.split('-')[0]
        baseline_latency = fetch_baseline_latency(model, args, baseline_latency_folder, use_inductor)
        if args.flops:
            baseline_latency = baseline_perf["flops"]

        kernels = collect_kernels(prefix / dir, model, args, use_inductor)
        kernels = list(filter(lambda x: x[1] >= min_acc, kernels))
        assert len(kernels) > 0, use_inductor
        
        x, y, flops, params, _, hash_value, latency, _ = zip(*kernels)

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
            c=color,
            linewidth=1.5,
            linestyle="-",
        )
        ax.scatter(
            latency[pareto_mask][id],
            y[pareto_mask][id],
            label=model2name[model],
            s=65,
            c=color,
            marker=marker,
        )
        ax.plot(
            [latency[pareto_mask][id][-1], baseline_latency],
            [y[pareto_mask][id][-1], reference_acc],
            c=color,
            linewidth=1.5,
            linestyle="--",
        )

        ax.scatter([baseline_latency], [reference_acc], s=65, marker=marker, facecolors='none', edgecolors=color, linewidths=1.5)

        min_speedup = baseline_latency / np.max(latency[pareto_mask])
        max_speedup = baseline_latency / np.min(latency[pareto_mask])
        min_speedups.append(min_speedup)
        max_speedups.append(max_speedup)
        
        logging.debug(f"{scenario} + {model2name[model]}: Speedup from {min_speedup:.2f} to {max_speedup:.2f}. Ratio for from {1 / min_speedup:.2f} to {1 / max_speedup:.2f}. ")

    logging.info(f"Average speedup for {scenario}: {geometric_mean(min_speedups):.2f}x-{geometric_mean(max_speedups):.2f}x")
        
    ax.set_title(scenario, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(left=0)
    ax.grid(linestyle='--')

if __name__ == "__main__":
    # Get Path
    args = parser()

    assert len(args.models) == len(args.dirs)
    
    fig, axs = plt.subplots(2, 3, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1, 1], 'wspace': 0.04, 'hspace': 0.06})
    fig.supxlabel("Inference time (ms)", fontsize=14)
    axs: np.ndarray

    for i, scenario in enumerate(Scenarios):
        draw(axs[0, i], args, scenario)
        
    for i, scenario in enumerate(Scenarios):
        # if i == 0: continue
        draw(axs[1, i], args, scenario, True)

    axs[0, 0].set_ylabel("Top-1 Accuracy", fontsize=14)
    axs[0, 0].set_xlim(right=130)
    axs[0, 0].xaxis.set_major_locator(MultipleLocator(20))
    axs[0, 0].set_xticklabels([])

    axs[0, 1].set_yticklabels([])
    axs[0, 1].set_xlim(right=21)
    axs[0, 1].xaxis.set_major_locator(MultipleLocator(5))
    axs[0, 1].set_xticklabels([])

    axs[0, 2].set_yticklabels([])
    axs[0, 2].set_xlim(right=2.9)
    axs[0, 2].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[0, 2].set_ylabel("TVM", fontsize=14, rotation=270, labelpad=20)
    axs[0, 2].yaxis.set_label_position("right")
    axs[0, 2].set_xticklabels([])
    
    axs[1, 0].set_title("")
    axs[1, 0].set_xlim(right=130)
    axs[1, 0].set_ylabel("Top-1 Accuracy", fontsize=14)
    axs[1, 0].xaxis.set_major_locator(MultipleLocator(20))

    axs[1, 1].set_title("")
    axs[1, 1].set_yticklabels([])
    axs[1, 1].set_xlim(right=21)
    axs[1, 1].xaxis.set_major_locator(MultipleLocator(5))

    axs[1, 2].set_title("")
    axs[1, 2].set_yticklabels([])
    axs[1, 2].set_xlim(right=2.9)
    axs[1, 2].xaxis.set_major_locator(MultipleLocator(0.5))
    axs[1, 2].set_ylabel("TorchInductor", fontsize=14, rotation=270, labelpad=20)
    axs[1, 2].yaxis.set_label_position("right")

    for ax in axs.flatten():
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.set_ylim(bottom=0.665)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=14, bbox_to_anchor=(0.5, 1.0))
    
    fig.subplots_adjust(top=0.85, bottom=0.12, left=0.06, right=0.97)
    fig.savefig(os.path.join(args.output, "imagenet-performance.pdf"))