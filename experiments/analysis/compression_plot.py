import easypyplot as epp

from plot_utils import *

def get_best_params(folder, model, args, min_acc):
    result = min([kernel for kernel in collect_kernels(folder, model, args) if kernel[1] > min_acc], key=lambda x:x[3])
    return result[3]

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
            'name': 'Ours',
            'data': [
                (name, get_best_params(folder, model, args, fetch_baseline_perf(model)["accuracy"] - args.max_acc_decrease) / fetch_baseline_perf(model)["params"])
                for name, model, folder in models
            ],
            'baseline': False,
            'text_mark': True
        },
    ]

    # Configurations
    name = 'compression'
    width = 0.4
    num_entries = len(entries)
    width_per_entry = width / num_entries
    linewidth = 0.6

    # Checks
    check_format(entries)
    baseline = check_baseline(entries, False)
    names, labels, bars = simplify(entries, baseline, True)
    num_groups = len(names)

    # Figures
    pp, fig = epp.pdf.plot_setup(os.path.join(args.output, f'{name}.pdf'), figsize=(6, 3.5), font='default')
    ax = fig.gca()

    # Draw bars
    epp.barchart.draw(ax, bars,
                      width=width,
                      linewidth=linewidth,
                      group_names=labels,
                      entry_names=names,
                      breakdown=False,
                      colors=[micro_nas_color],
                      legendloc='upper left',
                      xticklabelfontsize=9,
                      xticklabelrotation=20,
                    #   xticklabelrotationalignment='right'
                      )

    # Mark numbers
    text_numbers(ax, width, entries, bars, fontsize=8, extra_height=0.01)

    # Y axis
    ax.yaxis.grid(True)
    ax.set_ylabel('Model Size Ratio ×', multialignment='center', fontsize=11)
    ax.get_legend().remove()

    # Finish
    fig.tight_layout()
    fig.show()
    epp.pdf.plot_teardown(pp, fig)