import argparse
import dataclasses
from dataclasses import dataclass
import json
import os
import subprocess
import sys
import logging
from typing import Dict, Iterable, List, Optional

from common import RESNET_34_LAYERS_MODELS
from perf_torch import get_benchmark_output_path, _TORCH_TUNING_MODE


@dataclass
class TargetSpec:
    prefix: str
    baseline_dir: str
    device: str

    @property
    def working_dir(self) -> str:
        return os.path.join(self.prefix, self.baseline_dir)

    @staticmethod
    def from_dict(d: Dict[str, str]) -> "TargetSpec":
        prefix = d["prefix"]
        baseline_dir = d["baseline_dir"]
        device = d["device"]
        return TargetSpec(
            prefix=prefix,
            baseline_dir=baseline_dir,
            device=device,
        )


@dataclass
class BenchmarkConfig:
    model: str
    working_dir: str
    device: str
    channels_last: bool = False
    result_dir: Optional[str] = None
    batch_size: int = 1
    mode: str = _TORCH_TUNING_MODE
    disable_tf32: bool = False

    def to_args(self) -> List[str]:
        args = [
            "--model",
            self.model,
            "--working-dir",
            self.working_dir,
            "--device",
            self.device,
        ]
        if self.channels_last:
            args.append("--channels-last")
        if self.result_dir is not None:
            args.extend(["--result-dir", self.result_dir])
        if self.batch_size != 1:
            args.extend(["--batch-size", str(self.batch_size)])
        if self.mode != _TORCH_TUNING_MODE:
            args.extend(["--mode", self.mode])
        if self.disable_tf32:
            args.append("--disable-tf32")
        return args

    @property
    def output_path(self) -> str:
        return get_benchmark_output_path(
            model_name=self.model,
            target_type=self.device,
            mode=self.mode,
            working_dir=self.working_dir,
            kernels_dir=self.result_dir,
            batch_size=self.batch_size,
        )


def _make_grid(
    target: TargetSpec,
    model: str,
    kernels_dirs: List[str],
    channels_last: bool = False,
) -> List[BenchmarkConfig]:
    return [
        BenchmarkConfig(
            model=model,
            working_dir=target.working_dir,
            device=target.device,
            channels_last=channels_last,
            result_dir=os.path.join(target.prefix, k) if k is not None else None,
        )
        for k in [None] + kernels_dirs
    ]


def _make_grids(
    targets: List[TargetSpec],
    resnet_18_kernels_dirs: List[str],
    resnet_34_kernels_dirs: List[str],
    resnext_kernels_dirs: List[str],
    efficientnet_kernels_dirs: List[str],
    densenet_kernels_dirs: List[str],
    resnet_34_layers_kernels_dirs: List[str],
) -> List[BenchmarkConfig]:
    grids: List[BenchmarkConfig] = []
    for target in targets:
        grids.extend(_make_grid(
            target=target,
            model="torchvision/resnet18",
            kernels_dirs=resnet_18_kernels_dirs,
        ))
        grids.extend(_make_grid(
            target=target,
            model="torchvision/resnet34",
            kernels_dirs=resnet_34_kernels_dirs,
        ))
        grids.extend(_make_grid(
            target=target,
            model="torchvision/resnext29_2x64d",
            kernels_dirs=resnext_kernels_dirs,
        ))
        grids.extend(_make_grid(
            target=target,
            model="torchvision/efficientnet_v2_s",
            kernels_dirs=efficientnet_kernels_dirs,
        ))
        grids.extend(_make_grid(
            target=target,
            model="torchvision/densenet121",
            kernels_dirs=densenet_kernels_dirs,
        ))
        for layer in RESNET_34_LAYERS_MODELS:
            grids.extend(_make_grid(
                target=target,
                model=layer,
                kernels_dirs=resnet_34_layers_kernels_dirs,
                channels_last=True,
            ))
    return grids

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a custom configuration file with grids.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        default=False,
        help="Clear all instances, whether tuned or not. Use with caution!",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Do not run the tuning, just print the configurations that would be used.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config_file = args.config
    logging.info(f"Loading custom configuration from {config_file}")

    with open(config_file, "r") as f:
        config = json.load(f)
    targets = [TargetSpec.from_dict(t) for t in config["targets"]]
    kernels_dirs = config["kernels_dirs"]
    grids = _make_grids(
        targets=targets,
        resnet_18_kernels_dirs=kernels_dirs["resnet_18"],
        resnet_34_kernels_dirs=kernels_dirs["resnet_34"],
        resnext_kernels_dirs=kernels_dirs["resnext"],
        efficientnet_kernels_dirs=kernels_dirs["efficientnet"],
        densenet_kernels_dirs=kernels_dirs["densenet"],
        resnet_34_layers_kernels_dirs=kernels_dirs["resnet_34_layers"],
    )

    tasks: List[BenchmarkConfig] = []
    for config in grids:
        if os.path.exists(config.output_path):
            if args.clear:
                os.remove(config.output_path)
            else:
                continue
        tasks.append(config)

    if args.dry_run:
        logging.info("Dry run mode. The following configurations will be used:")
        for task in tasks:
            logging.info(f"  {task}")
        return 0

    for task in tasks:
        command = ["python", "perf_torch.py"] + task.to_args()
        logging.info(f"Running command: {' '.join(command)}")
        proc = subprocess.Popen(command)
        proc.communicate()
    logging.info("All tasks completed successfully.")

    return 0


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main(args))
