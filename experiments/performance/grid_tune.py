import argparse
import dataclasses
from dataclasses import dataclass
import logging
import multiprocessing as mp
import os
import queue
import sys
import time
from typing import Dict, Generator, List, Optional, Tuple, cast

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from MetaScheduleTuner import MetaScheduleTuner, ProgressDone, ProgressQueue, ProgressUpdate, TunerState, TuningConfig, parse_target, parse_target_preset, tune_e2e_mp_runner


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
    in_tuning_dirs: List[str] = []
    for kernels_grid in get_kernels_grids(GRIDS):
        model_tuner = kernels_grid.to_model_tuner()
        for config in kernels_grid:
            state = model_tuner.query_kernel_specific_tuner_state(config.kernels_dir)
            if state == TunerState.TUNED:
                tuned_trials += config.num_trials
            elif state == TunerState.TUNING:
                in_tuning_dirs.append(model_tuner.get_kernel_specific_tuner_working_dir(config.kernels_dir))
            else:
                assert state == TunerState.UNTUNED
            total_trials += config.num_trials
    has_tuning = len(in_tuning_dirs) > 0
    if has_tuning:
        logging.warning("WARNING: The following instances are still being tuned. Unless you want to overwrite them, please wait for them to finish.")
        for d in in_tuning_dirs:
            logging.warning(f"  {d}")
        logging.warning("If you do want to overwrite, please add option --overwrite.")
    return Stats(total_trials, tuned_trials, has_tuning)

@dataclass
class Tracker:
    index: int
    config: TuningConfig
    process: Optional[mp.Process] = None
    actual_trials: int = 0
    last_updated: float = dataclasses.field(default_factory=time.time)

    def start(self, mp_ctx, progress: "ProgressQueue[int]") -> None:
        assert self.process is None
        self.process = mp_ctx.Process(target=tune_e2e_mp_runner, args=(self.config, progress, self.index), daemon=True)
        self.process.start()
        self.update(0)
        logging.info(f"Started tuning {self.config}")

    def join(self) -> None:
        assert self.process is not None
        self.process.join()
        self.process = None

    def cancel_and_join(self) -> None:
        assert self.process is not None
        self.process.terminate()
        self.process.join()
        self.process = None

    # Return pbar update
    def update(self, trials: int) -> int:
        self.actual_trials += trials
        self.last_updated = time.time()
        return trials

    # Return pbar update
    def success(self) -> int:
        logging.info(f"Finished tuning {self.config}")
        # Ensure consistency
        return self.config.num_trials - self.actual_trials

    # Return pbar update
    def failure(self) -> int:
        logging.error(f"Error tuning {self.config}")
        # Ensure consistency
        return -self.actual_trials

    def timed_out(self, timeout: float) -> bool:
        return time.time() - self.last_updated > timeout

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallelism", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()

def main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    stats = _count_trials()
    logging.info(str(stats))
    if args.dry_run:
        return 0
    total_trials, tuned_trials = stats.total_trials, stats.tuned_trials
    objective_trials = total_trials - tuned_trials
    if stats.has_tuning:
        if args.overwrite:
            logging.info("Overwriting instances being tuned...")
        else:
            logging.error("Please wait for the instances being tuned to finish.")
            return 1

    parallelism: int = args.parallelism
    assert parallelism > 0
    timeout: float = args.timeout

    success = 0
    failure = 0
    timed_out = 0
    cancelled = 0
    mp_ctx = mp.get_context("spawn")
    with mp_ctx.Manager() as manager, tqdm(total=objective_trials, smoothing=0.0) as pbar, logging_redirect_tqdm():
        progress = cast("ProgressQueue[int]", manager.Queue())
        index = 0
        trackers: List[Tracker] = []
        for kernels_grid in get_kernels_grids(GRIDS):
            model_tuner = kernels_grid.to_model_tuner()
            for config in kernels_grid:
                state = model_tuner.query_kernel_specific_tuner_state(config.kernels_dir)
                if state == TunerState.TUNED:
                    continue
                trackers.append(Tracker(index=index, config=config))
                index += 1
        assert index == len(trackers)

        cursor = 0
        trackers_running: Dict[int, Tracker] = {}

        def enqueue_trackers() -> None:
            nonlocal cursor
            while len(trackers_running) < parallelism and cursor < len(trackers):
                tracker = trackers[cursor]
                cursor += 1
                trackers_running[tracker.index] = tracker
                tracker.start(mp_ctx, progress)

        def cancel_running_tracker(tracker: Tracker) -> None:
            tracker.cancel_and_join()
            del trackers_running[tracker.index]
            pbar.update(tracker.failure())

        enqueue_trackers()

        should_stop = False
        while len(trackers_running) > 0:
            try:
                p = progress.get(timeout=timeout)
                tracker = trackers_running[p.ctx]
                if isinstance(p, ProgressUpdate):
                    pbar.update(tracker.update(p.trials))
                elif isinstance(p, ProgressDone):
                    tracker.join()
                    del trackers_running[tracker.index]
                    if p.error:
                        pbar.update(tracker.failure())
                        failure += 1
                    else:
                        pbar.update(tracker.success())
                        success += 1
                else:
                    raise ValueError(f"Unexpected progress: {p}")
            except queue.Empty:
                # Some of the running tasks have timed out
                pass
            except KeyboardInterrupt:
                if should_stop:
                    logging.info("Terminating all running tasks...")
                    for index in list(trackers_running.keys()):
                        cancel_running_tracker(trackers_running[index])
                        cancelled += 1
                    break
                should_stop = True
                logging.info("Interrupted. Clearing all uninitiated tasks...")
                cancelled_trials = 0
                while cursor < len(trackers):
                    tracker = trackers[cursor]
                    cursor += 1
                    cancelled_trials += tracker.config.num_trials
                    cancelled += 1
                pbar.total -= cancelled_trials
                pbar.refresh()
                logging.info("Cleared. Press Ctrl+C again to exit.")
                logging.info("Pending tasks:")
                for tracker in trackers_running.values():
                    logging.info(f"  {tracker.config} ({tracker.actual_trials}/{tracker.config.num_trials})")

            # Check for timed out tasks
            for index in list(trackers_running.keys()):
                tracker = trackers_running[index]
                if tracker.timed_out(timeout):
                    cancel_running_tracker(tracker)
                    timed_out += 1

            # Enqueue new trackers
            enqueue_trackers()

        logging.info(f"All tuning finished: existing trials {tuned_trials}, objective_trials {objective_trials}, success {success}, failure {failure}, timed out {timed_out}, cancelled {cancelled}")

    return 0

if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main(args))
