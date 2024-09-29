from typing import Literal

import matplotlib.pyplot
import easypyplot as epp

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

def draw(ax, target):
    entries = [
        {
            'name': 'TVM',
            'data': read_entries(BASELINE_PATH, target),
            'baseline': True
        },
        {
            'name': 'Seq.1',
            'data': read_entries(SEQ_1_PATH, target),
            'baseline': False
        },
        {
            'name': 'Seq.2',
            'data': read_entries(SEQ_2_PATH, target),
            'baseline': False
        },
        {
            'name': 'Seq.3',
            'data': read_entries(SEQ_3_PATH, target),
            'baseline': False
        },
        {
            'name': 'Kernel 1',
            'data': read_entries(KERNEL_1_PATH, target),
            'baseline': False,
            'text_mark': True, 
            'offset': offset[target]['Kernel 1'],
        },
        {
            'name': 'Kernel 2',
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
    num_groups = len(names)


    # Draw bars
    epp.barchart.draw(ax, bars,
                      width=width,
                      linewidth=linewidth,
                      group_names=labels,
                      colors=[ansor_color, seq1_color, seq2_color, seq3_color, micro_nas_color, micro_nas_compress_color],
                      entry_names=names,
                      xticklabelfontsize=10,
                      breakdown=False)

    # Mark numbers
    text_numbers_custom(ax, width, entries, bars, fontsize=8)

    # Y axis
    ax.yaxis.grid(True)
    ax.set_ylabel(('GPU' if target == 'cuda' else 'CPU') + ' Speedup ×', multialignment='center', fontsize=10)
    YLIMS = {
        'cuda': 8,
        'llvm': 25,
    }
    ax.set_ylim(0, YLIMS[target])

if __name__ == '__main__':
    target = "cuda"
    # ResNet-34 individual layers

    name = f'kernel-performance'
    
    # Figures
    pp, fig = epp.pdf.plot_setup(f'analysis/results/{name}.pdf',
                                 font='default', figsize=(10.2, 4))

    ax1 = fig.add_subplot(2, 1, 1)
    draw(ax1, "llvm")
    ax1.xaxis.set_visible(False)
    
    ax2 = fig.add_subplot(2, 1, 2)
    draw(ax2, 'cuda')
    ax2.legend().remove()

    # Finish
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    fig.show()
    epp.pdf.plot_teardown(pp, fig)
