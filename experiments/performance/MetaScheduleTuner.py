# This file is adapted from apps/relax_examples/e2e_auto_tir.py in https://github.com/apache/tvm.
# See the license below.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import os
import sys
import csv
import argparse
from typing import Callable, Dict, Optional
import numpy as np
from importlib.util import spec_from_file_location, module_from_spec
from contextlib import ExitStack, nullcontext, redirect_stdout, redirect_stderr
from enum import Enum
from functools import wraps

import tvm
from tvm import relax, runtime, transform
from tvm.ir.module import IRModule
from tvm import meta_schedule as ms
from tvm.contrib.tar import tar

from common import get_specialized_model_name
from model import import_templated_model, substitute_kernels, construct_kernel_builder


class TunerState(Enum):
    UNTUNED = 0
    TUNING = 1
    TUNED = 2

class LogRedirector(ExitStack):
    def __init__(self, log_path: str) -> None:
        super().__init__()
        self._log_path = log_path
    def __enter__(self) -> 'LogRedirector':
        try:
            log = self.enter_context(open(self._log_path, "a"))
            self.enter_context(redirect_stdout(log))
            self.enter_context(redirect_stderr(log))
        except:
            if not self.__exit__(*sys.exc_info()):
                raise
        return self

class KernelSpecificTuner:
    def __init__(self, parent: 'MetaScheduleTuner', working_dir: str, relax_mod: IRModule) -> None:
        self._parent = parent
        self._target = self._parent._target
        self._working_dir = working_dir
        os.makedirs(self._working_dir, exist_ok=True)
        self._relax_mod = relax_mod
        self._db = None

    def get_relax_mod(self) -> IRModule:
        return self._relax_mod

    def optimize_model_before_tuning(self, show: bool = False) -> None:
        if show:
            print("Before optimization passes:")
            self._relax_mod.show()
        # Refer to tests/python/relax/test_meta_schedule_integration.py in in https://github.com/apache/tvm.
        seq = transform.Sequential([
            # Get TIR.
            relax.transform.DecomposeOpsForInference(),
            relax.transform.LegalizeOps(enable_warning=True),
            relax.transform.FoldConstant(),

            # Run fusion passes twice to ensure all possible fusions are done.
            # It is observed that some elementwise ops are not fused in the first pass.
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),

            # Remove redundant reshape ops, because they are no-ops.
            relax.transform.RewriteDataflowReshape(),
            # For debug purpose.
            relax.transform.AnnotateTIROpPattern(),
            # Clean up.
            relax.transform.DeadCodeElimination(),
        ])
        with self._target, transform.PassContext(opt_level=3):
            self._relax_mod = seq(self._relax_mod)
        if show:
            print("After optimization passes:")
            self._relax_mod.show()

    def tune(self, num_trials: int = 10, max_trials_per_task: Optional[int] = None, num_trials_per_iter: int = 64, on_eval: Optional[Callable[[], None]] = None) -> None:
        if max_trials_per_task is None:
            max_trials_per_task = num_trials
        self._db = ms.relax_integration.tune_relax(
            mod=self._relax_mod,
            target=self._target,
            params={},
            num_trials_per_iter=num_trials_per_iter,
            max_trials_per_task=max_trials_per_task,
            max_trials_global=num_trials,
            runner=self._parent.get_runner(on_eval),
            work_dir=self._working_dir,
        )

    def _get_exported_library_path(self) -> str:
        return os.path.join(self._working_dir, "kernels_tvm_tuned.tar")

    def build(self, show: bool = False) -> str:
        if self._db is None:
            with self._target, transform.PassContext(opt_level=3):
                executable = relax.build(self._relax_mod, target=self._target)
        else:
            with self._target, self._db, transform.PassContext(opt_level=3):
                relax_mod = relax.transform.MetaScheduleApplyDatabase(enable_warning=True)(self._relax_mod)
                if show:
                    print("After applying tuning database:")
                    relax_mod.show()
                with open(os.path.join(self._working_dir, "kernels_tvm_tuned.py"), "w") as out_file:
                    out_file.write(relax_mod.script(show_meta=True))
                executable = relax.build(relax_mod, target=self._target)
        exported_path = self._get_exported_library_path()
        executable.export_library(exported_path, tar)
        return exported_path

    def load(self) -> str:
        exported_path = self._get_exported_library_path()
        assert os.path.exists(exported_path), f"Cannot find {exported_path}. You need to tune and build the model first."
        return exported_path

    @staticmethod
    def benchmark_results_path(working_dir: str) -> str:
        return os.path.join(working_dir, "benchmark_results.csv")

    def get_benchmark_results_path(self) -> str:
        return self.benchmark_results_path(self._working_dir)

    @staticmethod
    def tuning_log_path(working_dir: str) -> str:
        return os.path.join(working_dir, "tuning.log")

    def get_tuning_log_path(self) -> str:
        return self.tuning_log_path(self._working_dir)

    @staticmethod
    def tuner_state(working_dir: str) -> TunerState:
        if os.path.exists(KernelSpecificTuner.benchmark_results_path(working_dir)):
            return TunerState.TUNED
        elif os.path.exists(KernelSpecificTuner.tuning_log_path(working_dir)):
            return TunerState.TUNING
        else:
            return TunerState.UNTUNED

    def get_tuner_state(self) -> TunerState:
        return self.tuner_state(self._working_dir)

    def redirect_log(self) -> LogRedirector:
        return LogRedirector(self.get_tuning_log_path())

    def measure_and_write(self, mod_path: str) -> runtime.module.BenchmarkResult:
        result = self._parent.measure(mod_path)
        out_path = self.get_benchmark_results_path()
        with open(out_path, "w") as out_file:
            writer = csv.writer(out_file)
            # write experiment parameters at the top as a record
            writer.writerow(["model", self._parent._model_name])
            writer.writerow(["input_shape", self._parent._input_shape])
            writer.writerow(["target", self._target])
            writer.writerow(["num_measurement_repeats", -1]) # We no longer use this field
            writer.writerow(["latency_mean", result.mean])
            for res in result.results:
                writer.writerow([str(res)])
        return result

def _perform_measurement(
    rt_mod: runtime.Module,
    device: runtime.ndarray.Device,
    input_data: Dict[str, np.ndarray],
    num_measurement_repeats: int = 3,
    num_measurements: int = 2,
) -> runtime.module.BenchmarkResult:
    print(f"num_measurement_repeats: {num_measurement_repeats}, num_measurements: {num_measurements}")
    vm = relax.VirtualMachine(rt_mod, device=device)
    nd_args = {k: runtime.ndarray.array(v, device) for k, v in input_data.items()}
    vm.save_function("main", "measure_func", **nd_args, include_return=False)
    evaluator = vm.time_evaluator(
        func_name="measure_func",
        dev=device,
        repeat=num_measurement_repeats,
        number=num_measurements,
        min_repeat_ms=500,
    )
    return evaluator()

_TOTAL_KERNELS_SUBSTITUTED = 0

class MetaScheduleTuner:
    def __init__(
        self,
        model: str,
        vanilla: bool,
        batch_size: int,
        target: tvm.target.Target,
        rpc_config: Optional[ms.runner.RPCConfig],
        working_dir: str,
    ) -> None:
        # Basic configurations.
        self._model_name = model
        self._target = target
        if self._target.attrs.get("mtriple", None) == "aarch64-linux-gnu":
            alloc_repeat = 3
        else:
            alloc_repeat = 1
        self._runner_config = {
            "evaluator_config": ms.runner.EvaluatorConfig(
                number=3,
                repeat=1,
                min_repeat_ms=100,
                enable_cpu_cache_flush=False,
            ),
            "alloc_repeat": alloc_repeat,
        }
        self._rpc_config = rpc_config
        self._specialized_dir_name = os.path.join(self._target.kind.name, get_specialized_model_name(self._model_name, batch_size, vanilla=vanilla))
        self._working_dir = os.path.join(working_dir, self._specialized_dir_name)

        # Load the model.
        self._templated_mod, self._all_mappings, self._input_shape = import_templated_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_relax"), self._model_name, batch_size, vanilla=vanilla)

        # Input shape for runtime.
        self.input_dtype = "float32"
        self.input_info = {"inp_0": self._input_shape}

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_kernel_specific_tuner_working_dir(self, kernels_dir: Optional[str]) -> str:
        if kernels_dir is None:
            return self._working_dir
        else:
            return os.path.join(kernels_dir, "perf", self._specialized_dir_name)

    def get_kernel_specific_tuner(self, kernels_dir: Optional[str]) -> KernelSpecificTuner:
        """Construct kernels according to the directory generated by Sampler.realize(). If the directory is None, return the original model."""
        working_dir = self.get_kernel_specific_tuner_working_dir(kernels_dir)
        kernel_builder = None
        if kernels_dir is not None:
            if os.path.exists(os.path.join(kernels_dir, "kernel_scheduler_dir")):
                kernels_file = os.path.join(kernels_dir, "kernel_scheduler_dir", "kernels_tvm.py")
            else:
                kernels_file = os.path.join(kernels_dir, "kernels_tvm.py")
            global _TOTAL_KERNELS_SUBSTITUTED
            spec = spec_from_file_location(f"kas_tvm_kernels_mod_{_TOTAL_KERNELS_SUBSTITUTED}", kernels_file)
            assert spec is not None, f"Failed to load module from {kernels_file}"
            _TOTAL_KERNELS_SUBSTITUTED += 1
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)
            kernel_builder = construct_kernel_builder(self._all_mappings, mod.weights, mod.build)
        relax_mod = substitute_kernels(self._templated_mod, kernel_builder)
        return KernelSpecificTuner(self, working_dir, relax_mod)

    def query_kernel_specific_tuner_state(self, kernels_dir: Optional[str]) -> TunerState:
        return KernelSpecificTuner.tuner_state(self.get_kernel_specific_tuner_working_dir(kernels_dir))

    def _get_num_workers(self) -> int:
        if self._rpc_config is not None:
            return self._rpc_config.count_num_servers(allow_missing=False)
        else:
            return 1

    @staticmethod
    def _wrap_run_evaluator(callback: Optional[Callable[[], None]]):
        def _wrap(f):
            if callback is not None:
                cb = callback
                @wraps(f)
                def _wrapped_f(*args, **kwargs):
                    ret = f(*args, **kwargs)
                    cb()
                    return ret
                return _wrapped_f
            else:
                return f
        return _wrap

    def get_runner(self, on_eval: Optional[Callable[[], None]] = None) -> ms.Runner:
        if self._rpc_config is not None:
            default_run_evaluator = ms.runner.rpc_runner.default_run_evaluator
            f_run_evaluator = self._wrap_run_evaluator(on_eval)(default_run_evaluator)
            runner = ms.runner.RPCRunner(
                rpc_config=self._rpc_config, max_workers=self._get_num_workers(), **self._runner_config, f_run_evaluator=f_run_evaluator
            )
        else:
            default_run_evaluator = ms.runner.local_runner.default_run_evaluator
            f_run_evaluator = self._wrap_run_evaluator(on_eval)(default_run_evaluator)
            runner = ms.runner.LocalRunner(**self._runner_config, f_run_evaluator=f_run_evaluator)
        return runner

    def _get_input_data(self) -> Dict[str, np.ndarray]:
        input_data: Dict[str, np.ndarray] = {}
        for input_name, input_shape in self.input_info.items():
            assert self.input_dtype.startswith("float"), "Only float input is supported."
            input_data[input_name] = np.random.uniform(size=input_shape).astype(self.input_dtype)
        return input_data

    def measure(self, mod_path: str) -> runtime.module.BenchmarkResult:
        input_data = self._get_input_data()
        dev_type = self._target.kind.name
        if self._rpc_config is not None:
            # Give longer timeout for RPC
            rpc_config = self._rpc_config._replace(session_timeout_sec=20)
            session = rpc_config.connect_server()
            session.upload(mod_path)
            filename = os.path.basename(mod_path)
            mod: runtime.Module = session.load_module(filename)
            dev = session.device(dev_type=dev_type, dev_id=0)
        else:
            mod = runtime.load_module(mod_path)
            dev = tvm.device(dev_type=dev_type, dev_id=0)
        result = _perform_measurement(mod, dev, input_data)
        return result

_cortex_a78_6_core = "llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a78 -mattr=+neon -num-cores=6"
_jetson_orin_nano_gpu = "cuda -arch=sm_87 -max_threads_per_block=1024 -max_num_threads=1024 -thread_warp_size=32 -max_shared_memory_per_block=49152 -registers_per_block=65536"
_x86_64_16_core = "llvm -mtriple=x86_64-pc-linux-gnu -num-cores=16"
_rtx_3060_laptop = "cuda -arch=sm_86 -max_threads_per_block=1024 -max_num_threads=1536 -thread_warp_size=32 -max_shared_memory_per_block=49152 -registers_per_block=65536"

PRESET_TARGETS = {
    "jetson_orin_nano-cpu": (_cortex_a78_6_core, _cortex_a78_6_core),
    "jetson_orin_nano-gpu": (_jetson_orin_nano_gpu, _cortex_a78_6_core),
    "x86_64-16_core_cpu": (_x86_64_16_core, _x86_64_16_core),
    "rtx_3060_laptop_gpu": (_rtx_3060_laptop, _x86_64_16_core),
}

def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model",
        type=str,
        default="resnet34layers",
        help="The model to tune.",
    )
    args.add_argument(
        "--vanilla",
        action="store_true",
        default=None,
        help="If enabled, do not perform substitution in models. This preserves the kernel parameters, including convolution window size, stride, etc.",
    )
    args.add_argument(
        "--no-vanilla",
        dest="vanilla",
        action="store_false",
    )
    args.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    args.add_argument(
        "--target",
        type=str,
        default=None,
    )
    args.add_argument(
        "--target-host",
        type=str,
        default=None,
        help="If target is GPU, host code is required to launch the kernel on the GPU, so a host-side target is also required."
    )
    args.add_argument(
        "--target-preset",
        type=str,
        default=None,
        help="Use a preset configuration for --target and --target-host."
    )
    args.add_argument(
        "--kernels-dir",
        type=str,
        default=None,
        help="The substitute kernel to be tuned."
    )
    args.add_argument(
        "--num-trials",
        type=int,
        default=4000,
    )
    args.add_argument(
        "--max-trials-per-task",
        type=int,
        default=None,
    )
    args.add_argument(
        "--num-trials-per-iter",
        type=int,
        default=64,
    )
    args.add_argument(
        "--load",
        action="store_true",
        default=False,
        help="Load the tuned model from the save instead of tuning it."
    )
    args.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite the existing tuning results."
    )
    args.add_argument(
        "--rpc",
        action="store_true",
        default=True,
        help="Use RPC. Please further set --rpc-host, --rpc-port, and --rpc-key."
    )
    args.add_argument(
        "--no-rpc",
        dest="rpc",
        action="store_false",
    )
    args.add_argument(
        "--rpc-host",
        type=str,
        default="127.0.0.1",
    )
    args.add_argument(
        "--rpc-port",
        type=int,
        default=9190,
    )
    args.add_argument(
        "--rpc-key",
        type=str,
        default="jetson-orin-nano",
    )
    args.add_argument(
        "--working-dir",
        type=str,
        default="./perf",
    )
    args.add_argument(
        "--redirect-log",
        action="store_true",
        default=True,
        help="Redirect the output to tuning.log in the kernel-specific working directory.",
    )
    args.add_argument(
        "--no-redirect-log",
        dest="redirect_log",
        action="store_false",
    )
    args.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Do not run the actual tuning, but show the model after optimization passes."
    )
    return args.parse_args()

def parse_target(target: Optional[str], target_host: Optional[str], target_preset: Optional[str]) -> tvm.target.Target:
    if target_preset is not None:
        assert (target is None) and (target_host is None), "target_preset is already specified."
        target, target_host = PRESET_TARGETS[target_preset]
    else:
        assert target is not None, "target must be specified."
        if target_host is None:
            target_host = target
    return tvm.target.Target(target, host=target_host)

def parse_rpc_config(rpc: bool, host: Optional[str], port: Optional[int], key: Optional[str], timeout: int = 5) -> Optional[ms.runner.RPCConfig]:
    if not rpc:
        return None
    assert (host is not None) and (port is not None) and (key is not None), "RPC is enabled, so host, port, and key must be specified."
    return ms.runner.RPCConfig(
        tracker_host=host,
        tracker_port=port,
        tracker_key=key,
        session_timeout_sec=timeout,
    )

def main():
    args = _parse_args()

    vanilla = args.vanilla if args.vanilla is not None else args.kernels_dir is None
    target = parse_target(args.target, args.target_host, args.target_preset)
    rpc_config = parse_rpc_config(args.rpc, args.rpc_host, args.rpc_port, args.rpc_key)

    tuner = MetaScheduleTuner(
        model=args.model,
        vanilla=vanilla,
        batch_size=args.batch_size,
        target=target,
        rpc_config=rpc_config,
        working_dir=args.working_dir,
    )
    kernels_tuner = tuner.get_kernel_specific_tuner(args.kernels_dir)

    # Check if the configuration has already been tuned.
    assert args.overwrite or (kernels_tuner.get_tuner_state() != TunerState.TUNED), f"This configuration has already been tuned. Use --overwrite to overwrite it."

    # Redirect output to a log file.
    redirect_log = kernels_tuner.redirect_log() if args.redirect_log else nullcontext()

    with redirect_log:
        if args.load:
            assert not args.dry_run
            print("Loading the tuned model...")
            mod_path = kernels_tuner.load()
        else:
            kernels_tuner.optimize_model_before_tuning(show=True)
            if args.dry_run:
                return
            kernels_tuner.tune(
                num_trials=args.num_trials,
                max_trials_per_task=args.max_trials_per_task,
                num_trials_per_iter=args.num_trials_per_iter,
            )
            mod_path = kernels_tuner.build(show=True)
        results = kernels_tuner.measure_and_write(mod_path)
        print("results:", results)

if __name__ == "__main__":
    main()
