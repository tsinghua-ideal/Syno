from typing import Literal

from matplotlib.patches import Patch
from matplotlib.axes import Axes
from easypyplot import pdf, barchart
from statistics import geometric_mean, mean

from plot_utils import *

NAMES_MAPPINGS = [
    ("L1", "conv_io64"),
    ("L7", "conv_i64_o128"),
    ("L8", "conv_io128"),
    ("L9", "residual_i64_o128"),
    ("L16", "conv_i128_o256"),
    ("L17", "conv_io256"),
    ("L18", "residual_i128_o256"),
    ("L29", "conv_i256_o512"),
    ("L30", "conv_io512"),
    ("L31", "residual_i256_o512"),
]

def read_entries(directory: str, target: Literal["cuda"] | Literal["llvm"]) -> list[tuple[str, float]]:
    return [
        (name, extract_latency(os.path.join(
            directory,
            target,
            "resnet34layers",
            f"{layer}-N=1",
            "benchmark_results.csv",
        )))
        for name, layer in NAMES_MAPPINGS
    ]

RESULT_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "results")
BASELINE_PATH = os.path.join(RESULT_PATH, "good_kernels/original")
SEQ_1_PATH = f"{RESULT_PATH}/nas-pte/seq_1/perf"
SEQ_2_PATH = f"{RESULT_PATH}/nas-pte/seq_2/perf"
SEQ_3_PATH = f"{RESULT_PATH}/nas-pte/seq_3/perf"
KERNEL_1_PATH = f"{RESULT_PATH}/resnet-good-kernels/0.6x/07889_15252107013978896537/perf"
KERNEL_2_PATH = f"{RESULT_PATH}/resnet-good-kernels/0.2x/07754_18091915762600937904/perf"

offset = {
    "llvm": {
        "Kernel 1": [(-0.18, 0.15), (-0.15, 0.15), (-0.15, 0.15), (0, 0.80), (-0.13, 1), (-0.13, 0.30), (0, 0.80), (-0.10, 2), (-0.13, 1.3), (0, 0.5)], 
        "Kernel 2": [(0, 0.15), (0, 0.15), (0, 0.15), (0, 2.0), (0, 0.15), (0, 0.15), (0, 2), (0, 0.15), (0, 0.15), (0, 2)]
    }, 
    "cuda": {
        "Kernel 1": [(-0.13, 0.8), (-0.13, 0.15), (-0.15, 0.15), (0, 0.30), (-0.13, 1), (-0.13, 0.30), (0, 0.50), (-0.13, 0.4), (-0.12, 0.15), (0, 0.50)], 
        "Kernel 2": [(0, 0.15), (0, 0.15), (0, 0.15), (0, 0.8), (0, 0.15), (0, 0.15), (0, 0.8), (0, 0.15), (0, 0.15), (0, 0.8)]
    }
}

def draw(ax: Axes, target):
    entries = [
        {
            'name': 'TVM',
            'data': read_entries(BASELINE_PATH, target),
            'baseline': True
        },
        {
            'name': 'NAS-PTE Seq 1',
            'data': read_entries(SEQ_1_PATH, target),
            'baseline': False
        },
        {
            'name': 'NAS-PTE Seq 2',
            'data': read_entries(SEQ_2_PATH, target),
            'baseline': False
        },
        {
            'name': 'NAS-PTE Seq 3',
            'data': read_entries(SEQ_3_PATH, target),
            'baseline': False
        },
        {
            'name': 'Syno Operator 1',
            'data': read_entries(KERNEL_1_PATH, target),
            'baseline': False,
            'text_mark': True, 
            'offset': offset[target]['Kernel 1'],
        },
        {
            'name': 'Syno Operator 2',
            'data': read_entries(KERNEL_2_PATH, target),
            'baseline': False,
            'text_mark': True, 
            'offset': offset[target]['Kernel 2'],
        },
    ]

    # Configurations
    width = 0.7
    num_entries = len(entries)
    width_per_entry = width / num_entries
    linewidth = 0.6

    # Checks
    check_format(entries)
    baseline = check_baseline(entries, True)
    names, labels, bars = simplify(entries, baseline, True)

    colors=[ansor_color, seq1_color, seq2_color, seq3_color, micro_nas_color, micro_nas_compress_color]
    hatchs=['o', 'x', '*', '-', '+', '|']

    # Draw bars
    barchart.draw(ax, bars,
                      width=width,
                      linewidth=linewidth,
                      group_names=labels,
                      colors=colors,
                      hatchs=hatchs, 
                      entry_names=names,
                      xticklabelfontsize=14,
                      breakdown=False)
    
    # print("Geomean speedup (Seq 1)", geometric_mean([x[1] for x in bars]))
    # print("Geomean speedup (Seq 2)", geometric_mean([x[2] for x in bars]))
    # print("Geomean speedup (Seq 3)", geometric_mean([x[3] for x in bars if x[3] > 0]))
    # print("Geomean speedup (Kernel 1)", geometric_mean([x[4] for x in bars]))
    # print("Geomean speedup (Kernel 2)", geometric_mean([x[5] for x in bars]))

    print("Geomean speedup (Kernel 1&2)", geometric_mean([max(x[4], x[5]) / max(x[1], x[2], x[3]) for x in bars]))

    # Mark numbers
    text_numbers_custom(ax, width, entries, bars, fontsize=11)

    # Y axis
    ax.yaxis.grid(True)
    YLIMS = {
        'cuda': 8.5,
        'llvm': 25,
    }
    ax.set_ylim(0, YLIMS[target])
    
    legend_elements = [
        Patch(facecolor=c, edgecolor='black', hatch=h, label=l)
        for c, h, l in zip(colors, hatchs, names)
    ]
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend().remove()

    ax.set_ylabel(('GPU' if target == 'cuda' else 'CPU') + ' Speedup', multialignment='center', fontsize=14)
    ax.grid(axis='y', linestyle='--')
    return legend_elements

if __name__ == '__main__':
    target = "cuda"
    # ResNet-34 individual layers

    name = f'kernel-performance'
    
    # Figures
    fig, axs = plt.subplots(2, 1, figsize=(14, 5), gridspec_kw={'hspace': 0.05, 'bottom': 0.08, 'top': 0.9, 'left': 0.05, 'right': 0.99})
    
    ax1 = axs[0]
    legend_elements = draw(ax1, "llvm")
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_label_coords(-0.03, 0.5)
    
    ax2 = axs[1]
    draw(ax2, 'cuda')
    ax2.yaxis.set_label_coords(-0.03, 0.5)

    fig.legend(handles=legend_elements, loc='upper center', ncol=6, fontsize=14, bbox_to_anchor=(0.5, 1.0))

    # Finish
    fig.savefig(os.path.join("analysis/results", f"{name}.pdf"))
