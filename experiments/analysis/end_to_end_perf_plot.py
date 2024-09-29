import easypyplot as epp

from plot_utils import *

def get_best_latency(folder, model, args, min_acc):
    result = min([kernel for kernel in collect_kernels(folder, model, args) if kernel[1] > min_acc], key=lambda x:x[6])
    return result[6]

def draw(ax, args):

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
            'name': 'Ours',
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
    
    # Draw bars
    epp.barchart.draw(ax, bars,
                      width=width,
                      linewidth=linewidth,
                      group_names=labels,
                      entry_names=names,
                      breakdown=False,
                      colors=[ansor_color, nas_pte_color, micro_nas_color],
                      xticklabelfontsize=8,
                      xticklabelrotation=20,
                    #   xticklabelrotationalignment='right'
                      )

    # Mark numbers
    text_numbers(ax, width, entries, bars, fontsize=7, extra_height=0.04 if args.gpu else 0.05)

    # Y axis
    ax.yaxis.grid(True)
    ax.set_ylabel(('GPU' if args.gpu else 'CPU') + ' Speedup ×', multialignment='center', fontsize=8)
    ax.set_ylim(0, 3.5 if args.gpu else 4.5)

if __name__ == '__main__':
    args = parser()
    
    name = 'end-to-end-performance'

    # Figures
    pp, fig = epp.pdf.plot_setup(os.path.join(args.output, f'{name}.pdf'), figsize=(5, 3), font='default')

    ax1 = fig.add_subplot(2, 1, 1)
    args.gpu = False
    draw(ax1, args)
    ax1.xaxis.set_visible(False)
    
    ax2 = fig.add_subplot(2, 1, 2)
    args.gpu = True
    draw(ax2, args)
    ax2.legend().remove()

    # Finish
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    fig.show()
    epp.pdf.plot_teardown(pp, fig)