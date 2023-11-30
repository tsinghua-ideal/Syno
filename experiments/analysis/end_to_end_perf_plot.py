import easypyplot as epp

from plot_utils import *

def get_best_latency(folder, model, args, min_acc):
    result = min([kernel for kernel in collect_kernels(folder, model, args) if kernel[1] > min_acc], key=lambda x:x[6])
    return result[6]

if __name__ == '__main__':
    args = parser()
    
    models = [
        ('ResNet-18', 'resnet18', 'results/good_kernels'), 
        ('ResNet-34', 'resnet34', 'results/cifar100-session-resnet34-reevaluate'), 
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
    name = 'end-to-end-performance'
    width = 0.6
    num_entries = len(entries)
    width_per_entry = width / num_entries
    linewidth = 0.7

    # Checks
    check_format(entries)
    baseline = check_baseline(entries, True)
    names, labels, bars = simplify(entries, baseline, True)
    num_groups = len(names)

    # Figures
    pp, fig = epp.pdf.plot_setup(os.path.join(args.output, f'{name}.pdf'), figsize=(5, 2), font='default')
    ax = fig.gca()

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
    text_numbers(ax, width, entries, bars, fontsize=7, extra_height=0.06)

    # Y axis
    ax.yaxis.grid(True)
    ax.set_ylabel('Speedup ×', multialignment='center', fontsize=8)
    ax.set_ylim(0,5.5)

    # Finish
    fig.tight_layout()
    fig.show()
    epp.pdf.plot_teardown(pp, fig)