import argparse
from contextlib import ExitStack
import dataclasses
from dataclasses import dataclass
import logging
import multiprocessing as mp
import os
import queue
import sys
import time
from typing import Dict, Generator, Iterable, List, Optional, Tuple, cast

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from MetaScheduleTuner import MetaScheduleTuner, ProgressDone, ProgressQueue, ProgressUpdate, TunerState, TuningConfig, parse_target, parse_target_preset, tune_e2e_mp_runner


@dataclass
class KernelsDirs(Iterable[Optional[str]]):
    content: List[Optional[str]]
    def __iter__(self) -> Generator[Optional[str], None, None]:
        for d in self.content:
            if d is None:
                yield None
            else:
                for f in os.listdir(d):
                    yield os.path.join(d, f)

@dataclass
class KernelsGrid(Iterable[TuningConfig]):
    model: str
    vanilla: bool
    target: str
    target_host: str
    trials: int
    kernels_dirs: Iterable[Optional[str]]

    def to_model_tuner(self) -> MetaScheduleTuner:
        return MetaScheduleTuner(
            model=self.model,
            vanilla=self.vanilla,
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
                vanilla=self.vanilla,
            )

@dataclass
class Grid:
    model: str
    vanilla: bool
    targets_and_trials: List[Tuple[str, int]]
    kernels_dirs: Iterable[Optional[str]]

    def __iter__(self) -> Generator[KernelsGrid, None, None]:
        for target_preset, trials in self.targets_and_trials:
            target, target_host = parse_target_preset(target_preset)
            yield KernelsGrid(self.model, self.vanilla, target, target_host, trials, self.kernels_dirs)

_RESNET_KERNELS_DIRS = KernelsDirs([
    "./results/resnet-good-kernels/0.2x",
    "./results/resnet-good-kernels/0.4x",
    "./results/resnet-good-kernels/0.5x",
    "./results/resnet-good-kernels/0.6x",
    "./results/resnet-good-kernels/0.7x",
    "./results/resnet-good-kernels/ablation",
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

_RESNET_34_LAYERS_KERNELS_DIRS = [
    # KAS
    "./results/resnet-good-kernels/0.6x/07889_15252107013978896537",
    "./results/resnet-good-kernels/0.2x/07754_18091915762600937904",
    # NAS-PTE
    "./results/nas-pte/seq_1",
    "./results/nas-pte/seq_2",
    "./results/nas-pte/seq_3",
]

_RESNET_34_LAYERS_NAMES = [
    # Same
    "conv_io64",
    "conv_io128",
    "conv_io256",
    "conv_io512",
    # Down
    "conv_i64_o128",
    "conv_i128_o256",
    "conv_i256_o512",
    # Residual
    "residual_i64_o128",
    "residual_i128_o256",
    "residual_i256_o512",
]

_RESNET_34_LAYERS_MODELS = [
    f"resnet34layers/{layer}"
    for layer in _RESNET_34_LAYERS_NAMES
]

def _make_targets_and_trials(trials: int) -> List[Tuple[str, int]]:
    return [
        ("a100_gpu", trials),
    ]

def _make_grid(model: str, trials: int, kernels_dirs: Iterable[Optional[str]]) -> Grid:
    return Grid(
        model,
        False,
        _make_targets_and_trials(trials),
        kernels_dirs,
    )

def _make_vanilla(grid: Grid, vanilla: bool = True) -> Grid:
    return Grid(
        grid.model,
        vanilla,
        grid.targets_and_trials,
        KernelsDirs([None]),
    )

def _make_both(model: str, trials: int, kernels_dirs: Iterable[Optional[str]], vanilla_for_vanilla: bool = True) -> List[Grid]:
    grid = _make_grid(model, trials, kernels_dirs)
    return [_make_vanilla(grid, vanilla=vanilla_for_vanilla), grid]

def _make_both_for_models(models: List[str], trials: int, kernels_dirs: Iterable[Optional[str]], vanilla_for_vanilla: bool = True) -> List[Grid]:
    return [
        grid
        for model in models
        for grid in _make_both(model, trials, kernels_dirs, vanilla_for_vanilla=vanilla_for_vanilla)
    ]

GRIDS = [
    *_make_both(
        "torchvision/resnet18",
        10000,
        _RESNET_KERNELS_DIRS,
    ),
    *_make_both(
        "torchvision/resnet34",
        10000,
        _RESNET_KERNELS_DIRS,
    ),
    *_make_both(
        "torchvision/resnext29_2x64d",
        15000,
        _RESNEXT_KERNELS_DIRS,
    ),
    *_make_both(
        "torchvision/efficientnet_v2_s",
        30000,
        _EFFICIENTNET_KERNELS_DIRS,
    ),
    *_make_both(
        "torchvision/densenet121",
        50000,
        _DENSENET_KERNELS_DIRS,
    ),
    *_make_both_for_models(
        _RESNET_34_LAYERS_MODELS,
        4000,
        _RESNET_34_LAYERS_KERNELS_DIRS,
        vanilla_for_vanilla=False,
    ),
]

def get_kernels_grids(grids: List[Grid]) -> List[KernelsGrid]:
    return [k for g in grids for k in g]

@dataclass
class Tracker:
    index: int
    config: TuningConfig
    process: Optional[mp.Process] = None
    actual_trials: int = 0
    last_updated: float = dataclasses.field(default_factory=time.time)

    @property
    def total_trials(self) -> int:
        return self.config.num_trials

    def start(self, mp_ctx, progress: "ProgressQueue[int]") -> None:
        assert self.process is None
        self.process = mp_ctx.Process(target=tune_e2e_mp_runner, args=(self.config, progress, self.index), daemon=True)
        self.process.start()
        self.update(0)

    def is_active(self) -> bool:
        return self.process is not None

    def join(self) -> None:
        assert self.process is not None
        self.process.join()
        self.process = None

    def cancel_and_join(self) -> None:
        assert self.process is not None
        self.process.terminate()
        self.process.join()
        self.process = None

    def update(self, trials: int) -> None:
        self.actual_trials += trials
        self.last_updated = time.time()

    def is_timed_out(self, timeout: float) -> bool:
        return time.time() - self.last_updated > timeout

class TrackerManager(ExitStack):
    """Lifecycle of a tracker:
    1. Pending
    2. Running (from pending trackers)
    3. Done (can be either success or failure, from running trackers)
    4. Skipped (upon graceful shutdown, from pending trackers)
    5. Aborted (upon forced shutdown or timeout, from running trackers)
    """
    def __init__(self, trackers: List[Tracker], parallelism: int, timeout: float) -> None:
        super().__init__()

        self._trackers = trackers
        self._parallelism = parallelism
        self._timeout = timeout
        self._mp_ctx = mp.get_context("spawn")

        # State
        self.trackers_running: Dict[int, Tracker] = {}
        self._cursor = 0

        # Resources
        self._manager = self._mp_ctx.Manager()
        self._progress = cast("ProgressQueue[int]", self._manager.Queue())
        objective_trials = sum(t.total_trials for t in trackers)
        self._pbar = tqdm(total=objective_trials, smoothing=0.0)

        # Stats
        self.success = 0
        self.failure = 0
        self.skipped = 0
        self.cancelled = 0
        self.timed_out = 0

    def __enter__(self) -> "TrackerManager":
        super().__enter__()
        self.enter_context(self._manager)
        self.enter_context(self._pbar)
        self.enter_context(logging_redirect_tqdm())
        return self

    def __exit__(self, *exc_info) -> None:
        self.cancel_all_running()
        self.skip_all_pending()
        super().__exit__(*exc_info)

    def _update_pbar(self, delta: int, delta_total: int = 0) -> None:
        self._pbar.total += delta_total
        self._pbar.update(delta)
        self._pbar.refresh()

    def _next_pending_tracker(self) -> Optional[Tracker]:
        if self._cursor < len(self._trackers):
            tracker = self._trackers[self._cursor]
            self._cursor += 1
            return tracker
        return None

    def _run_tracker(self, tracker: Tracker) -> None:
        """Pending -> Running."""
        self.trackers_running[tracker.index] = tracker
        tracker.start(self._mp_ctx, self._progress)
        logging.info(f"Started tuning {tracker.config}")

    def launch_trackers(self) -> None:
        """Pending -> Running. Launch as many trackers as possible."""
        while len(self.trackers_running) < self._parallelism:
            tracker = self._next_pending_tracker()
            if tracker is None:
                break
            self._run_tracker(tracker)

    def _abort_tracker(self, tracker: Tracker) -> None:
        """Running -> Aborted."""
        tracker.cancel_and_join()
        del self.trackers_running[tracker.index]
        self._update_pbar(
            delta=0,
            delta_total=tracker.actual_trials - tracker.total_trials,
        )

    def cancel_all_running(self) -> None:
        """Running -> Aborted (Cancelled)."""
        for tracker in list(self.trackers_running.values()):
            assert tracker.is_active()
            self._abort_tracker(tracker)
            self.cancelled += 1

    def skip_all_pending(self) -> None:
        """Pending -> Skipped."""
        while True:
            tracker = self._next_pending_tracker()
            if tracker is None:
                break
            assert not tracker.is_active()
            assert tracker.actual_trials == 0
            self._update_pbar(0, delta_total=-tracker.total_trials)
            self.skipped += 1

    def timeout_tracker(self, tracker: Tracker) -> None:
        """Running -> Aborted (Timed out)."""
        assert tracker.is_active()
        self._abort_tracker(tracker)
        logging.error(f"Timed out: {tracker.config}")
        self.timed_out += 1

    def on_update(self, tracker: Tracker, trials: int) -> None:
        """Running -> Running."""
        tracker.update(trials)
        self._update_pbar(delta=trials)

    def _on_done(self, tracker: Tracker) -> None:
        tracker.join()
        del self.trackers_running[tracker.index]
        self._update_pbar(
            delta=0,
            # Ensure consistency.
            delta_total=tracker.actual_trials - tracker.total_trials,
        )

    def on_success(self, tracker: Tracker) -> None:
        """Running -> Done (Success)."""
        self._on_done(tracker)
        logging.info(f"Finished tuning: {tracker.config}")
        self.success += 1

    def on_failure(self, tracker: Tracker) -> None:
        """Running -> Done (Failure)."""
        self._on_done(tracker)
        logging.error(f"Failed to tune: {tracker.config}")
        self.failure += 1

    def check_timed_out_trackers(self) -> None:
        for tracker in list(self.trackers_running.values()):
            if tracker.is_timed_out(self._timeout):
                self.timeout_tracker(tracker)

    def run(self) -> None:
        """The main loop."""

        # Fill the pipeline.
        self.launch_trackers()

        should_stop = False
        while len(self.trackers_running) > 0:
            try:
                p = self._progress.get(timeout=self._timeout)
                tracker = self.trackers_running[p.ctx]
                if isinstance(p, ProgressUpdate):
                    self.on_update(tracker, p.trials)
                elif isinstance(p, ProgressDone):
                    if p.error:
                        self.on_failure(tracker)
                    else:
                        self.on_success(tracker)
                else:
                    raise ValueError(f"Unexpected progress: {p}")
            except queue.Empty:
                # Some of the running tasks have timed out
                pass
            except KeyboardInterrupt:
                if should_stop:
                    logging.info("Terminating all running tasks...")
                    self.cancel_all_running()
                    break
                should_stop = True
                logging.info("Interrupted. Clearing all uninitiated tasks...")
                self.skip_all_pending()
                logging.info("Cleared. Press Ctrl+C again to exit.")
                logging.info("Pending tasks:")
                for tracker in self.trackers_running.values():
                    logging.info(f"  {tracker.config} ({tracker.actual_trials}/{tracker.total_trials})")

            # Check for timed out tasks
            self.check_timed_out_trackers()

            # Enqueue new trackers
            self.launch_trackers()

        logging.info(f"All tuning finished: success {self.success}, failure {self.failure}, skipped {self.skipped}, timed out {self.timed_out}, cancelled {self.cancelled}")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallelism", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()

def main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    index = 0
    trackers: List[Tracker] = []
    in_tuning_dirs: List[str] = []
    for kernels_grid in get_kernels_grids(GRIDS):
        model_tuner = kernels_grid.to_model_tuner()
        for config in kernels_grid:
            state = model_tuner.query_kernel_specific_tuner_state(config.kernels_dir)
            if state == TunerState.TUNED:
                continue
            if state == TunerState.TUNING:
                in_tuning_dirs.append(model_tuner.get_kernel_specific_tuner_working_dir(config.kernels_dir))
            trackers.append(Tracker(index=index, config=config))
            index += 1
    assert index == len(trackers)

    has_tuning = len(in_tuning_dirs) > 0
    if has_tuning:
        logging.warning("WARNING: The following instances are still being tuned. Unless you want to overwrite them, please wait for them to finish.")
        for d in in_tuning_dirs:
            logging.warning(f"  {d}")
        logging.warning("If you do want to overwrite, please add option --overwrite.")

    if args.dry_run:
        return 0

    if has_tuning:
        if args.overwrite:
            logging.info("Overwriting instances being tuned...")
        else:
            logging.error("Please wait for the instances being tuned to finish.")
            return 1

    parallelism: int = args.parallelism
    assert parallelism > 0
    timeout: float = args.timeout
    assert timeout > 0

    with TrackerManager(trackers, parallelism=parallelism, timeout=timeout) as manager:
        manager.run()

    return 0

if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main(args))
