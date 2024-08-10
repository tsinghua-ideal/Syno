import argparse
import concurrent.futures
from dataclasses import dataclass
import logging
import os
import queue
from typing import Generator, List, Tuple

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from MetaScheduleTuner import MetaScheduleTuner, Progress, ProgressDone, ProgressUpdate, TunerState, TuningConfig, parse_target, parse_target_preset, tune_e2e_mp


@dataclass
class KernelsDirs:
    content: List[str]
    def __iter__(self):
        for d in self.content:
            for f in os.listdir(d):
                yield os.path.join(d, f)

@dataclass
class KernelsGrid:
    model: str
    target: str
    target_host: str
    trials: int
    kernels_dirs: KernelsDirs

    def to_model_tuner(self) -> MetaScheduleTuner:
        return MetaScheduleTuner(
            model=self.model,
            vanilla=False,
            target=parse_target(self.target, self.target_host),
        )

    def __iter__(self) -> Generator[TuningConfig, None, None]:
        for k in self.kernels_dirs:
            yield TuningConfig(
                model=self.model,
                target=self.target,
                target_host=self.target_host,
                kernels_dir=k,
                num_trials=self.trials,
            )

@dataclass
class Grid:
    model: str
    targets_and_trials: List[Tuple[str, int]]
    kernels_dirs: KernelsDirs

    def __iter__(self) -> Generator[KernelsGrid, None, None]:
        for target_preset, trials in self.targets_and_trials:
            target, target_host = parse_target_preset(target_preset)
            yield KernelsGrid(self.model, target, target_host, trials, self.kernels_dirs)

_RESNET_KERNELS_DIRS = KernelsDirs([
    "./results/resnet-good-kernels/0.2x",
    "./results/resnet-good-kernels/0.4x",
    "./results/resnet-good-kernels/0.5x",
    "./results/resnet-good-kernels/0.6x",
    "./results/resnet-good-kernels/0.7x",
])

_RESNEXT_KERNELS_DIRS = KernelsDirs([
    "./results/resnext-good-kernels",
])

_EFFICIENTNET_KERNELS_DIRS = KernelsDirs([
    "./results/efficientnet-good-kernels",
])

_DENSENET_KERNELS_DIRS = KernelsDirs([
    "./results/densenet-good-kernels",
])

def _make_targets_and_trials(trials: int) -> List[Tuple[str, int]]:
    return [
        ("jetson_orin_nano-gpu", trials),
        ("jetson_orin_nano-cpu", trials),
    ]

GRIDS = [
    Grid(
        "torchvision/resnet18",
        _make_targets_and_trials(10000),
        _RESNET_KERNELS_DIRS,
    ),
    Grid(
        "torchvision/resnet34",
        _make_targets_and_trials(10000),
        _RESNET_KERNELS_DIRS,
    ),
    Grid(
        "torchvision/resnext29_2x64d",
        _make_targets_and_trials(15000),
        _RESNEXT_KERNELS_DIRS,
    ),
    Grid(
        "torchvision/efficientnet_v2_s",
        _make_targets_and_trials(30000),
        _EFFICIENTNET_KERNELS_DIRS,
    ),
    Grid(
        "torchvision/densenet121",
        _make_targets_and_trials(50000),
        _DENSENET_KERNELS_DIRS,
    ),
]

def get_kernels_grids(grids: List[Grid]) -> List[KernelsGrid]:
    return [k for g in grids for k in g]

@dataclass
class Stats:
    total_trials: int
    tuned_trials: int
    has_tuning: bool

def _count_trials() -> Stats:
    """Returns total trials and tuned trials."""
    total_trials = 0
    tuned_trials = 0
    in_tuning_configs: List[TuningConfig] = []
    for kernels_grid in get_kernels_grids(GRIDS):
        model_tuner = kernels_grid.to_model_tuner()
        for config in kernels_grid:
            state = model_tuner.query_kernel_specific_tuner_state(config.kernels_dir)
            if state == TunerState.TUNED:
                tuned_trials += config.num_trials
            elif state == TunerState.TUNING:
                in_tuning_configs.append(config)
            else:
                assert state == TunerState.UNTUNED
            total_trials += config.num_trials
    has_tuning = len(in_tuning_configs) > 0
    if has_tuning:
        logging.warning("WARNING: The following instances are still being tuned. Unless you want to overwrite them, please wait for them to finish.")
        for config in in_tuning_configs:
            logging.warning(f"  {model_tuner.get_kernel_specific_tuner_working_dir(config.kernels_dir)}")
        logging.warning("If you do want to overwrite, please add option --overwrite.")
    return Stats(total_trials, tuned_trials, has_tuning)

def start_tuning(config: TuningConfig, progress: queue.Queue[Progress], index: int, timeout: float) -> int:
    """This runs on a worker thread."""
    logging.info(f"Started tuning {config}")
    tune_e2e_mp(
        config,
        progress,
        ctx=index,
        timeout=timeout,
    )
    logging.info(f"Finished tuning {config}")
    return index

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallelism", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()

def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    stats = _count_trials()
    logging.info(str(stats))
    if args.dry_run:
        return
    total_trials, tuned_trials = stats.total_trials, stats.tuned_trials
    if stats.has_tuning:
        if args.overwrite:
            logging.info("Overwriting instances being tuned...")
        else:
            logging.error("Please wait for the instances being tuned to finish.")
            return

    parallelism: int = args.parallelism
    assert parallelism > 0
    timeout: float = args.timeout

    success = 0
    failure = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor, tqdm(total=total_trials, initial=tuned_trials) as pbar, logging_redirect_tqdm():
        progress: queue.Queue[Progress] = queue.Queue()
        index = 0
        futures = []
        for kernels_grid in get_kernels_grids(GRIDS):
            model_tuner = kernels_grid.to_model_tuner()
            for config in kernels_grid:
                state = model_tuner.query_kernel_specific_tuner_state(config.kernels_dir)
                if state == TunerState.TUNED:
                    continue
                future = executor.submit(start_tuning, config, progress, index, timeout)
                futures.append(future)
                index += 1
        assert index == len(futures)
        remaining = index

        # Collect progress
        while remaining > 0:
            p = progress.get()
            if isinstance(p, ProgressUpdate):
                pbar.update(p.trials)
            elif isinstance(p, ProgressDone):
                remaining -= 1
                index = futures[p.ctx].result()
                assert index == p.ctx
                if p.error:
                    failure += 1
                else:
                    success += 1
            else:
                raise ValueError(f"Unexpected progress: {p}")

        logging.info(f"All tuning finished: existing {tuned_trials}, success {success}, failure {failure}, total {success + failure}")

if __name__ == "__main__":
    args = _parse_args()
    main(args)
