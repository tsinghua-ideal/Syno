from typing import Literal

from matplotlib.patches import Patch
from matplotlib.axes import Axes
from easypyplot import barchart
from statistics import geometric_mean

from plot_utils import *
from pathlib import Path
import math

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

def read_entries(directory: Path | str, target: Literal["cuda"] | Literal["llvm"] | Literal["inductor-cuda"] | Literal["inductor-cpu"], inductor_baseline=False) -> list[tuple[str, float]]:
    file_name = "/benchmark_results.csv" if not target.startswith("inductor-") else ".txt"
    if inductor_baseline:
        file_name = "-orig" + file_name
    divisor = 1e3 if target.startswith("inductor-") else 1
    return [
        (name, extract_latency(os.path.join(
            directory,
            target,
            "resnet34layers",
            f"{layer}-N=1{file_name}"
        ))/divisor)
        for name, layer in NAMES_MAPPINGS
    ]

RESULT_PATH = "results"
BASELINE_PATH = os.path.join(RESULT_PATH, "good_kernels/original")
SEQ_1_PATH = f"{RESULT_PATH}/nas-pte/seq_1/perf"
SEQ_2_PATH = f"{RESULT_PATH}/nas-pte/seq_2/perf"
SEQ_3_PATH = f"{RESULT_PATH}/nas-pte/seq_3/perf"
KERNEL_1_PATH = f"{RESULT_PATH}/resnet-good-kernels/0.6x/07889_15252107013978896537/perf"
KERNEL_2_PATH = f"{RESULT_PATH}/resnet-good-kernels/0.2x/07754_18091915762600937904/perf"

offset = {
    "Mobile CPU": {
        "Kernel 1": [(-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2)], 
        "Kernel 2": [(0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2)], 
        "Kernel 1 - TorchInductor": [(-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2)], 
        "Kernel 2 - TorchInductor": [(0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2)]
    }, 
    "Mobile GPU": {
        "Kernel 1": [(-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2)], 
        "Kernel 2": [(0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2)], 
        "Kernel 1 - TorchInductor": [(-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2)], 
        "Kernel 2 - TorchInductor": [(0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2)]
    }, 
    "A100": {
        "Kernel 1": [(-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2)], 
        "Kernel 2": [(0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2)], 
        "Kernel 1 - TorchInductor": [(-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2), (-0.02, 0.2)], 
        "Kernel 2 - TorchInductor": [(0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2), (0.02, 0.2)]
    }
}

def draw(ax: Axes, scenario: str):

    if scenario == "Mobile CPU": 
        prefix = Path(os.getcwd())
        target = "llvm"
        max_y = 11
    elif scenario == "Mobile GPU": 
        prefix = Path(os.getcwd())
        target = "cuda"
        max_y = 6.5
    else: 
        prefix = Path("/cephfs/suzhengyuan/kas-a100-benchmark-results")
        target = "cuda"
        max_y = 4.5
        
    baseline_name = "TVM"

    entries = [
        {
            'name': baseline_name,
            'data': read_entries(prefix / BASELINE_PATH, target),
            'baseline': True
        },
        {
            'name': f'NAS-PTE Seq 1 w/ {baseline_name}',
            'data': read_entries(prefix / SEQ_1_PATH, target),
            'baseline': False
        },
        {
            'name': f'NAS-PTE Seq 2 w/ {baseline_name}',
            'data': read_entries(prefix / SEQ_2_PATH, target),
            'baseline': False
        },
        {
            'name': f'NAS-PTE Seq 3 w/ {baseline_name}',
            'data': read_entries(prefix / SEQ_3_PATH, target),
            'baseline': False
        },
        {
            'name': f'Syno Operator 1 w/ {baseline_name}',
            'data': read_entries(prefix / KERNEL_1_PATH, target),
            'baseline': False,
            'text_mark': True, 
            'offset': offset[scenario]['Kernel 1'],
        },
        {
            'name': f'Syno Operator 2 w/ {baseline_name}',
            'data': read_entries(prefix / KERNEL_2_PATH, target),
            'baseline': False,
            'text_mark': True, 
            'offset': offset[scenario]['Kernel 2'],
        },
    ]
    divisor = [[1.0 for _ in entry['data']] for entry in entries]

    baseline_name = 'TorchInductor'
    target = "inductor-cuda" if scenario != "Mobile CPU" else "inductor-cpu"
    extended_entries = [
        {
            'name': "_placeholder",
            'data': [ (name, math.inf) for name, _ in NAMES_MAPPINGS ], 
            'baseline': False,
            'text_mark': False
        },
        {
            'name': baseline_name,
            'data': read_entries(prefix / BASELINE_PATH, target, inductor_baseline=True),
            'baseline': False,
            'text_mark': False, 
        },
        {
            'name': f'NAS-PTE Seq 1 w/ {baseline_name}',
            'data': read_entries(prefix / SEQ_1_PATH, target),
            'baseline': False
        },
        {
            'name': f'NAS-PTE Seq 2 w/ {baseline_name}',
            'data': read_entries(prefix / SEQ_2_PATH, target),
            'baseline': False
        },
        {
            'name': f'NAS-PTE Seq 3 w/ {baseline_name}',
            'data': read_entries(prefix / SEQ_3_PATH, target),
            'baseline': False
        },
        {
            'name': 'Syno Operator 1 w/ TorchInductor',
            'data': read_entries(prefix / KERNEL_1_PATH, target),
            'baseline': False,
            'text_mark': True, 
            'offset': offset[scenario]['Kernel 1 - TorchInductor'],
        },
        {
            'name': 'Syno Operator 2 w/ TorchInductor',
            'data': read_entries(prefix / KERNEL_2_PATH, target),
            'baseline': False,
            'text_mark': True, 
            'offset': offset[scenario]['Kernel 2 - TorchInductor'],
        },
    ]
    entries.extend(extended_entries)
    divisor.extend([
        [baseline_perf / new_baseline_perf for (_, baseline_perf), (_, new_baseline_perf) in zip(entries[0]['data'], extended_entries[1]['data'])]
        for _ in extended_entries])
    divisor[1] = [1.0 for _ in divisor[1]]

    # Configurations
    width = 0.7
    num_entries = len(entries)
    width_per_entry = width / num_entries
    linewidth = 0.6

    # Checks
    check_format(entries)
    baseline = check_baseline(entries, True)
    names, labels, bars = simplify(entries, baseline, True)

    colors=[
        ansor_color, seq1_color, seq2_color, seq3_color, op1_color, op2_color, 
        ansor_color, 
        inductor_color, seq1_inductor_color, seq2_inductor_color, seq3_inductor_color, op1_inductor_color, op2_inductor_color
    ]

    # Draw bars
    barchart.draw(ax, bars,
                      width=width,
                      linewidth=linewidth,
                      group_names=labels,
                      colors=colors,
                    #   hatchs=hatchs, 
                      entry_names=names,
                      xticklabelfontsize=14,
                      breakdown=False)

    print("Speedup in %s: TVM - %.4f x; TorchInductor - %.4f x" % (scenario, 
        geometric_mean([max(x[4], x[5]) / max(x[1], x[2], x[3]) for x in bars]), 
        geometric_mean([max(x[11], x[12]) / max(x[8], x[9], x[10]) for x in bars])))

    # Mark numbers
    text_numbers_custom(ax, width, entries, bars, max_y, fontsize=9, rotation=90, divisor=divisor)

    # Y axis
    ax.yaxis.grid(True)
    ax.set_ylim(0, max_y)
    
    legend_elements = [
        Patch(facecolor=c, edgecolor='black', label=l)
        for c, l in zip(colors, names) if not l.startswith("_")
    ]
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.get_legend().remove()

    ax.set_ylabel(scenario, fontsize=14, rotation=270, labelpad=20)
    ax.yaxis.set_label_position("right")
    ax.grid(axis='y', linestyle='--')
    # ax.set_yscale('symlog', linthresh=linthreshy)
    return legend_elements

if __name__ == '__main__':
    # ResNet-34 individual layers

    name = f'kernel-performance'
    
    # Figures
    fig, axs = plt.subplots(3, 1, figsize=(14, 8), gridspec_kw={'hspace': 0.05, 'bottom': 0.06, 'top': 0.88, 'left': 0.08, 'right': 0.97})
    
    ax1 = axs[0]
    draw(ax1, "Mobile CPU")
    ax1.xaxis.set_visible(False)
    draw(axs[1], "Mobile GPU")
    legend_elements = draw(axs[2], "A100")

    fig.supylabel("Speedup", fontsize=16)
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, fontsize=12, bbox_to_anchor=(0.5, 1.0))

    # Finish
    fig.savefig(os.path.join("analysis/results", f"{name}.pdf"))
