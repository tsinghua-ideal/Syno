import argparse
import os
import subprocess

layers_same = [
    "conv_io64",
    "conv_io128",
    "conv_io256",
    "conv_io512",
]

layers_down = [
    "conv_i64_o128",
    "conv_i128_o256",
    "conv_i256_o512",
]

layers_residual = [
    "residual_i64_o128",
    "residual_i128_o256",
    "residual_i256_o512",
]

orig = [
    None,
]

kas = [
    "results/resnet-good-kernels/0.6x/07889_15252107013978896537",
    "results/resnet-good-kernels/0.2x/07754_18091915762600937904",
]

pte = [
    "results/nas-pte/seq_1",
    "results/nas-pte/seq_2",
    "results/nas-pte/seq_3",
]

def run(layer: str, kernel: str | None, *additional) -> None:
    log_file = f"tuning_{layer}.log"
    if kernel is not None:
        log_file = os.path.join(kernel, log_file)
    if os.path.exists(log_file):
        print(f"Skipping {layer} for {kernel}")
        return
    command = [
        "python",
        "MetaScheduleTuner.py",
        "--model",
        f"resnet34layers/{layer}",
    ]
    if kernel is not None:
        command += [
            "--kernels-dir",
            kernel,
        ]
    else:
        command += [
            "--vanilla",
            "False",
        ]
    command += [str(x) for x in additional]
    print(f"Running command: {' '.join(command)}")
    print(f"Logging to {log_file}")
    with open(log_file, "w") as f:
        subprocess.run(command, stdout=f, stderr=f)

def grid(layers: list[str], kernels: list[str | None], *additional) -> None:
    for layer in layers:
        for kernel in kernels:
            run(layer, kernel, *additional)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--same",
        default=False,
        action="store_true",
    )
    args.add_argument(
        "--down",
        default=False,
        action="store_true",
    )
    args.add_argument(
        "--residual",
        default=False,
        action="store_true",
    )
    args.add_argument(
        "--orig",
        default=False,
        action="store_true",
    )
    args.add_argument(
        "--kas",
        default=False,
        action="store_true",
    )
    args.add_argument(
        "--pte",
        default=False,
        action="store_true",
    )
    args = args.parse_args()

    layers = []
    if args.same:
        layers += layers_same
    if args.down:
        layers += layers_down
    if args.residual:
        layers += layers_residual
    kernels = []
    if args.orig:
        kernels += orig
    if args.kas:
        kernels += kas
    if args.pte:
        kernels += pte

    print(f"Layers: {layers}")
    print(f"Kernels: {kernels}")
    print(f"Grid size: {len(layers) * len(kernels)}")

    grid(layers, kernels)
