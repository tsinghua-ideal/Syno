import numpy as np
import os
import argparse
from argparse import Namespace
import logging
from types import ModuleType
from typing import List, Optional
import importlib
from matplotlib import pyplot as plt

import tvm
from tvm import relay, auto_scheduler, te, topi
from tvm.relay import data_dep_optimization as ddo
import tvm.relay.testing
from tvm.contrib import graph_executor
from tvm.contrib.utils import tempdir


def parse_configs():
    parser = argparse.ArgumentParser(description="KAS TVM-based performance tuner")

    # Device config.
    parser.add_argument("--target-string", type=str, default="llvm -mtriple=aarch64-linux-gnu -mattr=+neon")
    parser.add_argument("--device-key", type=str, default="jetson-orin-nano")
    parser.add_argument("--rpc-host", type=str, default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=9190)

    # Misc.
    parser.add_argument("--working-dir", type=str, default="./measurements")

    # Tuning.
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--min-repeat-ms", type=int, default=200)

    args = parser.parse_args()

    # Print
    args_str = "\n  > ".join([f"{k}: {v}" for k, v in vars(args).items()])
    logging.info(f"Execution arguments: \n  > {args_str}")

    return args

class OperatorTuner:
    def __init__(self, args: Namespace, workload_mod: ModuleType) -> None:
        self.args = args
        self.target = tvm.target.Target(args.target_string)
        self.dtype = "float32"
        self.kernel_name = workload_mod.kernel_name
        self.log_file = os.path.join(args.working_dir, f"{self.kernel_name}.log")
        self.workload_mod = workload_mod
        self.task = auto_scheduler.SearchTask(
            func=self.kernel_name, args=(), target=self.target
        )

    def _log_file_exists(self) -> bool:
        return os.path.exists(self.log_file)

    def compute_dag(self) -> auto_scheduler.ComputeDAG:
        return self.task.compute_dag

    def load_costs(self) -> List[float]:
        """Costs in all trials."""
        if not self._log_file_exists():
            return []
        return [
            float(sum(measure.costs) / len(measure.costs))
            for _, measure in auto_scheduler.load_records(self.log_file)
        ]

    def tune(self, trials: Optional[int] = None) -> float:
        if trials is None:
            trials = self.args.trials

        if self._log_file_exists():
            cost_model = auto_scheduler.XGBModel()
            logging.info(f"Loading previous schedule trials from {self.log_file}...")
            cost_model.update_from_file(self.log_file)
            search_policy = auto_scheduler.SketchPolicy(
                self.task,
                cost_model,
                init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(self.log_file)]
            )
        else:
            cost_model = auto_scheduler.XGBModel()
            search_policy = auto_scheduler.SketchPolicy(self.task, cost_model)

        logging.debug("Begin tuning...")
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=trials,  # change this to 20000 to achieve the best performance
            runner=auto_scheduler.RPCRunner(
                key=self.args.device_key,
                host=self.args.rpc_host,
                port=self.args.rpc_port,
                timeout=self.args.timeout,
                repeat=self.args.repeat,
                min_repeat_ms=self.args.min_repeat_ms,
                enable_cpu_cache_flush=True,
            ),
            measure_callbacks=[auto_scheduler.RecordToFile(self.log_file)],
        )

        self.task.tune(tune_option, search_policy=search_policy)

    def evaluate(self) -> float:
        """Return the best end-to-end latency in seconds."""
        logging.debug(f"Started evaluation of {self.kernel_name}.")

        # Compile with the history best
        logging.debug("Compiling...")
        sch, args = self.task.apply_best(self.log_file)
        func = tvm.build(sch, args, self.target, name=self.kernel_name)

        # Export library
        tmp = tempdir()
        filename = f"{self.kernel_name}.tar"
        func.export_library(tmp.relpath(filename))

        # Upload module to device
        logging.debug("Uploading...")
        remote = auto_scheduler.utils.request_remote(self.args.device_key, self.args.rpc_host, self.args.rpc_port, timeout=self.args.timeout)
        remote.upload(tmp.relpath(filename))
        rlib = remote.load_module(filename)

        # Create graph executor
        dev = remote.cpu()
        in_0_tvm = tvm.nd.array(np.random.uniform(size=self.workload_mod.input_shape).astype(self.dtype), dev)
        in_other_tvm = [tvm.nd.array(np.random.uniform(size=shape).astype(self.dtype), dev) for shape in self.workload_mod.weights_shapes]
        out_tvm = tvm.nd.array(np.zeros(self.workload_mod.output_shape, dtype=self.dtype), dev)

        # Evaluate
        logging.debug("Evaluate inference time cost...")
        timed_rlib = rlib.time_evaluator(rlib.entry_name, dev, number=10, min_repeat_ms=500)
        cost = timed_rlib(in_0_tvm, *in_other_tvm, out_tvm).mean
        logging.debug("Latency: %g ms/op" % (1000 * cost))

        return cost

    def filtered_costs(self) -> List[float]:
        costs = self.load_costs()

        # Compute Q1, Q3, and IQR
        Q1 = np.percentile(costs, 25)
        Q3 = np.percentile(costs, 75)
        IQR = Q3 - Q1
        
        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter out the outliers
        filtered_costs = [cost for cost in costs if lower_bound <= cost <= upper_bound]
        
        return filtered_costs

    def plot_cost_vs_trial(self) -> None:
        costs = self.filtered_costs()

        # Compute the running minimum for each trial
        running_min = [costs[0]]
        for cost in costs[1:]:
            running_min.append(min(running_min[-1], cost))

        trials = list(range(len(costs)))

        # Plot the data
        plt.scatter(trials, costs, c='blue', label='Trials')
        plt.step(trials, running_min, c='red', where='post', label='Min Cost')

        plt.xlabel('Trial')
        plt.ylabel('Cost')
        plt.legend()
        plt.title('Cost versus Trial')
        plt.grid(True)
        plt.show()

class Tuner:
    def __init__(self, args: Optional[Namespace] = None) -> None:
        if args is None:
            args = parse_configs()
        self.args = args

    def _load_module(self, module_path: str) -> ModuleType:
        spec = importlib.util.spec_from_file_location(f"workload_{hash(module_path)}", module_path)
        workload_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(workload_mod)
        return workload_mod

    def get_operator_tuner(self, module_path: str) -> OperatorTuner:
        workload_mod = self._load_module(module_path)
        return OperatorTuner(self.args, workload_mod)
