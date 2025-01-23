from easypyplot import pdf, barchart
from matplotlib.patches import Patch
from statistics import geometric_mean
from matplotlib.ticker import MultipleLocator
from matplotlib.axes import Axes

from matplotlib.pyplot import rcParams
rcParams.update({'font.size': 14})

from plot_utils import *

def get_best_latency(folder, model, args, min_acc):
    result = min([kernel for kernel in collect_kernels(folder, model, args) if kernel[1] > min_acc], key=lambda x:x[6])
    return result[6]

def draw(ax: Axes, args):

    models = [
        ('ResNet-18', 'resnet18', 'results/resnet-good-kernels'), 
        ('ResNet-34', 'resnet34', 'results/resnet-good-kernels'), 
        ('DenseNet-121', 'densenet121', 'results/densenet-good-kernels'), 
        ('ResNeXt-29', 'resnext29_2x64d', 'results/resnext-good-kernels'), 
        ('EfficientNet-V2-S', 'efficientnet_v2_s', 'results/efficientnet-good-kernels')
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
            'name': 'Syno',
            'data': [
                (name, get_best_latency(folder, model, args, fetch_baseline_perf(model)["accuracy"] - args.max_acc_decrease))
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

    colors=[ansor_color, nas_pte_color, micro_nas_color]
    hatchs=['*', '\\']
    
    # Draw bars
    barchart.draw(ax, bars,
                      width=width,
                      linewidth=linewidth,
                      group_names=labels,
                      entry_names=names,
                      breakdown=False,
                      colors=colors,
                      hatchs=hatchs, 
                      xticklabelfontsize=14,
                      xticklabelrotation=20,
                    #   xticklabelrotationalignment='right'
                      )

    # Mark numbers
    text_numbers(ax, width, entries, bars, fontsize=14, extra_height=0.04 if args.gpu else 0.05)
    print("Geomean speedup", geometric_mean([x[1] for x in bars]))

    # Y axis
    ax.grid(axis='y', linestyle='--')
    ax.set_ylabel(('GPU' if args.gpu else 'CPU') + ' Speedup', multialignment='center', fontsize=14)
    ax.set_ylim(0, 3.5 if args.gpu else 4.5)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=14)
    legend_elements = [
        Patch(facecolor=c, edgecolor='black', hatch=h, label=l)
        for c, h, l in zip(colors, hatchs, names)
    ]
    ax.legend(handles=legend_elements, fontsize=14)

if __name__ == '__main__':
    args = parser()
    
    name = 'end-to-end-performance'

    # Figures
    fig, axs = plt.subplots(2, 1, figsize=(7, 4), gridspec_kw={'hspace': 0.05, 'bottom': 0.22, 'top': 0.98, 'left': 0.08, 'right': 0.99})

    ax1 = axs[0]
    args.gpu = False
    draw(ax1, args)
    ax1.xaxis.set_visible(False)
    
    ax2 = axs[1]
    args.gpu = True
    draw(ax2, args)
    ax2.legend().remove()

    # Finish
    fig.savefig(os.path.join(args.output, f"{name}.pdf"))