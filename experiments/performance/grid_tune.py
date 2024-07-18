import argparse
import concurrent.futures
from dataclasses import dataclass
import logging
import multiprocessing as mp
import os
import queue
from typing import Iterable, List, Optional, Tuple, Union

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from MetaScheduleTuner import KernelSpecificTuner, MetaScheduleTuner, TunerState, parse_rpc_config, parse_target


@dataclass
class KernelsDirs:
    content: List[str]
    def __iter__(self):
        for d in self.content:
            for f in os.listdir(d):
                yield os.path.join(d, f)

@dataclass
class Grid:
    model: str
    targets_and_trials: List[Tuple[str, int]]
    kernels_dirs: KernelsDirs

RPC_CONFIG = parse_rpc_config(True, "127.0.0.1", 9190, "jetson-orin-nano")

def get_model_tuner(model: str, target: str) -> MetaScheduleTuner:
    target = parse_target(None, None, target)
    return MetaScheduleTuner(
        model=model,
        vanilla=False,
        batch_size=1,
        target=target,
        rpc_config=RPC_CONFIG,
        working_dir="./perf",
    )

@dataclass
class SerializableInstance:
    model: str
    target: str
    trials: int
    kernel_path: str
    def instantiate(self) -> KernelSpecificTuner:
        tuner = get_model_tuner(self.model, self.target)
        return tuner.get_kernel_specific_tuner(self.kernel_path)

@dataclass
class Instance:
    tuner: MetaScheduleTuner
    target: str
    trials: int
    kernel_path: str
    def query_state(self) -> TunerState:
        return self.tuner.query_kernel_specific_tuner_state(self.kernel_path)
    def get_working_dir(self) -> str:
        return self.tuner.get_kernel_specific_tuner_working_dir(self.kernel_path)
    def to_serializable(self) -> SerializableInstance:
        return SerializableInstance(
            model=self.tuner.model_name,
            target=self.target,
            trials=self.trials,
            kernel_path=self.kernel_path,
        )

@dataclass
class KernelsGrid:
    tuner: MetaScheduleTuner
    target: str
    trials: int
    kernels_dirs: KernelsDirs
    def __iter__(self):
        for k in self.kernels_dirs:
            yield Instance(self.tuner, self.target, self.trials, k)

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

def get_model_tuners(grids: List[Grid]) -> Iterable[KernelsGrid]:
    for grid in grids:
        for target, trials in grid.targets_and_trials:
            tuner = get_model_tuner(grid.model, target)
            kernels_dirs = grid.kernels_dirs
            yield KernelsGrid(tuner, target, trials, kernels_dirs)

def get_instances(grids: List[Grid]) -> Iterable[Instance]:
    model_tuners = get_model_tuners(GRIDS)
    for model_tuner in model_tuners:
        yield from model_tuner

@dataclass
class Stats:
    total_trials: int
    tuned_trials: int
    has_tuning: bool

def _count_trials() -> Stats:
    """Returns total trials and tuned trials."""
    kernel_tuners = get_instances(GRIDS)
    total_trials = 0
    tuned_trials = 0
    tuning_instances: List[Instance] = []
    for kernel_tuner in kernel_tuners:
        state = kernel_tuner.query_state()
        if state == TunerState.TUNED:
            tuned_trials += kernel_tuner.trials
        elif state == TunerState.TUNING:
            tuning_instances.append(kernel_tuner)
        else:
            assert state == TunerState.UNTUNED
        total_trials += kernel_tuner.trials
    has_tuning = len(tuning_instances) > 0
    if has_tuning:
        logging.warning("WARNING: The following instances are still being tuned. Unless you want to overwrite them, please wait for them to finish.")
        for instance in tuning_instances:
            logging.warning(f"  {instance.get_working_dir()}")
        logging.warning("If you do want to overwrite, please add option --overwrite.")
    return Stats(total_trials, tuned_trials, has_tuning)

def run_tuning(instance: SerializableInstance, progress: "mp.Queue[Optional[int]]"):
    """This runs on another process."""
    tuner = instance.instantiate()
    counter = 0
    def on_eval():
        nonlocal counter
        counter += 1
        progress.put(1)
    with tuner.redirect_log():
        tuner.optimize_model_before_tuning(show=True)
        tuner.tune(
            num_trials=instance.trials,
            on_eval=on_eval,
        )
        mod_path = tuner.build(show=True)
        results = tuner.measure_and_write(mod_path)
        print("results:", results)
    # Ensure consistency
    progress.put(instance.trials - counter)
    # Notify the main process that the task is done
    progress.put(None)

@dataclass
class ProgressUpdate:
    trials: int
@dataclass
class ProgressDone:
    index: int
Progress = Union[ProgressUpdate, ProgressDone]

def start_tuning(instance: SerializableInstance, index: int, progress: queue.Queue[Progress]) -> int:
    """This runs on a worker thread."""
    mp_progress: mp.Queue[Optional[int]] = mp.Queue()
    proc = mp.Process(target=run_tuning, args=(instance, mp_progress))
    proc.start()
    logging.info(f"Started tuning {instance}")
    trials_counter = 0
    while True:
        p = mp_progress.get()
        if p is None:
            # Done
            break
        trials_counter += p
        progress.put(ProgressUpdate(p))
    mp_progress.close()
    proc.join()
    logging.info(f"Finished tuning {instance}")
    # Notify the main thread that the task is done
    progress.put(ProgressDone(index))
    return index

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallelism", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser.parse_args()

def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)

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

    mp.set_start_method("spawn")

    kernel_tuners = get_instances(GRIDS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as executor, tqdm(total=total_trials, initial=tuned_trials) as pbar, logging_redirect_tqdm():
        progress: queue.Queue[Progress] = queue.Queue()
        index = 0
        futures = []
        for kernel_tuner in kernel_tuners:
            state = kernel_tuner.query_state()
            if state == TunerState.TUNED:
                continue
            instance = kernel_tuner.to_serializable()
            future = executor.submit(start_tuning, instance, index, progress)
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
                index = futures[p.index].result()
                assert index == p.index
            else:
                raise ValueError(f"Unexpected progress: {p}")

        logging.info("All tuning finished.")

if __name__ == "__main__":
    main()
