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
            "perf",
            target,
            "resnet34layers",
            f"{layer}-N=1",
            "benchmark_results.csv",
        )))
        for name, layer in NAMES_MAPPINGS
    ]

BASELINE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "performance")
SEQ_1_PATH = f"{BASELINE_PATH}/results/nas-pte/seq_1"
SEQ_2_PATH = f"{BASELINE_PATH}/results/nas-pte/seq_2"
SEQ_3_PATH = f"{BASELINE_PATH}/results/nas-pte/seq_3"
KERNEL_1_PATH = f"{BASELINE_PATH}/results/resnet-good-kernels/0.6x/07889_15252107013978896537"
KERNEL_2_PATH = f"{BASELINE_PATH}/results/resnet-good-kernels/0.2x/07754_18091915762600937904"

if __name__ == '__main__':
    target = "cuda"
    # ResNet-34 individual layers
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
            'offset': [(0, 0) for _ in range(len(NAMES_MAPPINGS))],
        },
        {
            'name': 'Kernel 2',
            'data': read_entries(KERNEL_2_PATH, target),
            'baseline': False,
            'text_mark': True, 
            'offset': [(0, 0) for _ in range(len(NAMES_MAPPINGS))],
        },
    ]

    # Configurations
    name = f'kernel-performance-{target}'
    width = 0.7
    num_entries = len(entries)
    width_per_entry = width / num_entries
    linewidth = 0.6

    # Checks
    check_format(entries)
    baseline = check_baseline(entries, True)
    names, labels, bars = simplify(entries, baseline, True)
    num_groups = len(names)

    # Figures
    pp, fig = epp.pdf.plot_setup(f'analysis/results/{name}.pdf',
                                 font='default', figsize=(10.2, 2.4))
    ax = fig.gca()

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
    ax.set_ylabel('Speedup ×', multialignment='center', fontsize=10)
    YLIMS = {
        'cuda': 8,
        'llvm': 25,
    }
    ax.set_ylim(0, YLIMS[target])

    # Finish
    fig.tight_layout()
    fig.show()
    epp.pdf.plot_teardown(pp, fig)
