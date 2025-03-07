from easypyplot import pdf, barchart
from matplotlib.patches import Patch
from statistics import geometric_mean
from matplotlib.ticker import MultipleLocator
from matplotlib.axes import Axes

from matplotlib.pyplot import rcParams
rcParams.update({'font.size': 14})

from plot_utils import *
from pathlib import Path
import math


plot_configs: dict[str, dict[str, float]] = {
    "Mobile CPU": {
        "extra_height": 0.05, 
        "max_y": 4.5
    }, 
    "Mobile GPU": {
        "extra_height": 0.04, 
        "max_y": 3.5
    }, 
    "A100": {
        "extra_height": 0.08, 
        "max_y": 4.5
    }, 
}
def get_best_latency(folder, model, args, min_acc, use_inductor=False):
    result = min([kernel for kernel in collect_kernels(folder, model, args, use_inductor) if kernel[1] > min_acc], key=lambda x:x[6])
    return result[6]

def draw(ax: Axes, args, scenario: str):

    if scenario == "Mobile CPU": 
        prefix = Path(os.getcwd()) / "results"
        args.gpu = False
    elif scenario == "Mobile GPU": 
        prefix = Path(os.getcwd()) / "results"
        args.gpu = True
    else: 
        prefix = Path("/cephfs/suzhengyuan/kas-a100-benchmark-results") / "results"
        args.gpu = True
        args.baseline_latency_folder = "/cephfs/shared/Syno/perf"

    models = [
        ('ResNet-18', 'resnet18', prefix / 'resnet-good-kernels'), 
        ('ResNet-34', 'resnet34', prefix / 'resnet-good-kernels'), 
        ('DenseNet-121', 'densenet121', prefix / 'densenet-good-kernels'), 
        ('ResNeXt-29', 'resnext29_2x64d', prefix / 'resnext-good-kernels'), 
        ('EfficientNet-V2-S', 'efficientnet_v2_s', prefix / 'efficientnet-good-kernels')
    ]
    
    entries = [
        {
            'name': 'TVM',
            'data': [
                (name, fetch_baseline_latency(model, args))
                for name, model, _ in models
            ],
            'baseline': True
        },
        {
            'name': 'Syno w/ TVM',
            'data': [
                (name, get_best_latency(folder, model, args, fetch_baseline_perf(model)["accuracy"] - args.max_acc_decrease))
                for name, model, folder in models
            ],
            'baseline': False,
            'text_mark': True
        },
        {
            'name': "_placeholder1",
            'data': [
                (name, math.inf)
                for name, _, _ in models
            ],
            'baseline': False,
            'text_mark': False
        },
        {
            'name': "_placeholder2",
            'data': [
                (name, math.inf)
                for name, model, _ in models
            ],
            'baseline': False,
            'text_mark': False
        },
        {
            'name': 'TorchInductor',
            'data': [
                (name, fetch_torchinductor_latency(model, args))
                for name, model, _ in models
            ],
            'baseline': False,
            'text_mark': False
        },
        {
            'name': 'Syno w/ TorchInductor',
            'data': [
                (name, get_best_latency(folder, model, args, fetch_baseline_perf(model)["accuracy"] - args.max_acc_decrease, use_inductor=True))
                for name, model, folder in models
            ],
            'baseline': False,
            'text_mark': True
        },
    ]

    # Configurations
    width = 0.6
    num_entries = len(entries)
    width_per_entry = width / num_entries
    linewidth = 0.7

    # Checks
    check_format(entries)
    baseline = check_baseline(entries, True)
    names, labels, bars = simplify(entries, baseline, True)
    num_groups = len(names)

    colors=[ansor_color, op1_color, nas_pte_color, nas_pte_color, inductor_color, op1_inductor_color]
    
    # Draw bars
    barchart.draw(ax, bars,
                      width=width,
                      linewidth=linewidth,
                      group_names=labels,
                      entry_names=names,
                      breakdown=False,
                      colors=colors,
                    #   hatchs=hatchs, 
                      xticklabelfontsize=14,
                      xticklabelrotation=20,
                    #   xticklabelrotationalignment='right'
                      )

    # Mark numbers
    num_groups = len(entries[0]['data'])

    for j in range(num_groups):
        value = bars[j][1]
        ax.text(j - width / 2 + width_per_entry * (1 + 0.5), value + plot_configs[scenario]["extra_height"], '{:.2f}×'.format(value), fontsize=12, ha='center')
        
    for j in range(num_groups):
        value = bars[j][5] / bars[j][4]
        ax.text(j - width / 2 + width_per_entry * (5 + 0.5), bars[j][5] + plot_configs[scenario]["extra_height"], '{:.2f}×'.format(value), fontsize=12, ha='center')

    print("Speedup of %s: TVM - %.4f x; TorchInductor - %.4f x" % (scenario, geometric_mean([x[1] for x in bars]), geometric_mean([x[5] / x[4] for x in bars])))

    # Y axis
    ax.grid(axis='y', linestyle='--')
    ax.set_ylabel(scenario, multialignment='center', fontsize=14)
    ax.set_ylim(0, plot_configs[scenario]["max_y"])
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(axis='both', which='major', labelsize=14)
    legend_elements = [
        Patch(facecolor=c, edgecolor='black', label=l)
        for c, l in zip(colors, names) if "placeholder" not in l
    ]
    return legend_elements

if __name__ == '__main__':
    args = parser()
    
    name = 'end-to-end-performance'

    # Figures
    fig, axs = plt.subplots(3, 1, figsize=(7, 5.5), gridspec_kw={'hspace': 0.05, 'bottom': 0.16, 'top': 0.93, 'left': 0.13, 'right': 0.99})
    fig.supylabel("Speedup")

    for i, scenario in enumerate(Scenarios):
        legends = draw(axs[i], args, scenario)

    fig.legend(handles=legends, loc='upper center', ncol=4, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    axs[0].xaxis.set_visible(False)
    axs[0].get_legend().remove()
    axs[1].get_legend().remove()
    axs[2].get_legend().remove()

    # Finish
    fig.savefig(os.path.join(args.output, f"{name}.pdf"))