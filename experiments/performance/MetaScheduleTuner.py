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
import csv
import argparse
from distutils.util import strtobool
from typing import Dict, List, Optional
import numpy as np
import importlib

import tvm
from tvm import relax, runtime, transform
from tvm.ir.module import IRModule
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc

from common import get_specialized_model_name
from model import import_templated_model, substitute_kernels, construct_kernel_builder


class KernelSpecificTuner:
    def __init__(self, parent: 'MetaScheduleTuner', working_dir: os.PathLike, relax_mod: IRModule) -> None:
        self.parent = parent
        self.working_dir = working_dir
        self.relax_mod = relax_mod
        self.db = None

    def get_relax_mod(self) -> IRModule:
        return self.relax_mod

    def optimize_model_before_tuning(self, show: bool = False) -> None:
        seq = transform.Sequential([
            relax.transform.DecomposeOpsForInference(),
            relax.transform.LegalizeOps(enable_warning=True),
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FoldConstant(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
        ])
        with transform.PassContext(opt_level=3):
            self.relax_mod = seq(self.relax_mod)
        if show:
            print("After optimization passes:")
            self.relax_mod.show()

    def tune(self, num_trials: int = 10) -> None:
        self.db = ms.relax_integration.tune_relax(
            mod=self.relax_mod,
            target=self.parent.target,
            params=None,
            num_trials_per_iter=64,
            max_trials_per_task=num_trials,
            max_trials_global=num_trials,
            runner=self.parent._get_runner(),
            work_dir=self.working_dir,
        )

    def build(self, show: bool = False) -> relax.Executable:
        if self.db is None:
            with transform.PassContext(opt_level=3):
                executable = relax.build(self.relax_mod, target=self.parent.target)
        else:
            with self.parent.target, self.db, transform.PassContext(opt_level=3):
                relax_mod = relax.transform.MetaScheduleApplyDatabase(enable_warning=True)(self.relax_mod)
                if show:
                    print("After applying tuning database:")
                    relax_mod.show()
                with open(os.path.join(self.working_dir, "kernels_tvm_tuned.py"), "w") as out_file:
                    out_file.write(relax_mod.script(show_meta=True))
                executable = relax.build(relax_mod, target=self.parent.target)
        from tvm.contrib.tar import tar
        executable.export_library(os.path.join(self.working_dir, "kernels_tvm_tuned.tar.gz"), tar)
        return executable

    def measure(self, executable: relax.Executable) -> runtime.module.BenchmarkResult:
        result = self.parent._measure(executable)
        out_path = os.path.join(self.working_dir, "benchmark_results.csv")
        with open(out_path, "w") as out_file:
            writer = csv.writer(out_file)
            # write experiment parameters at the top as a record
            writer.writerow(["model", self.parent.model_name])
            writer.writerow(["input_shape", self.parent.input_shape])
            writer.writerow(["target", self.parent.target])
            writer.writerow(["num_measurement_repeats", self.parent.num_measurement_repeats])
            writer.writerow(["latency_mean", result.mean])
            for res in result.results:
                writer.writerow([str(res)])
        return result

_TOTAL_KERNELS_SUBSTITUTED = 0

class MetaScheduleTuner:
    def __init__(
        self,
        model: str,
        vanilla: bool,
        batch_size: int,
        target: str,
        rpc_host: Optional[str] = "127.0.0.1",
        rpc_port: Optional[int] = 9190,
        rpc_key: Optional[str] = "jetson-orin-nano",
        working_dir: os.PathLike = "./measurements",
        rpc_timeout_sec: int = 180,
        num_measurement_repeats: int = 3,
        num_measurements: int = 2
    ) -> None:
        # Basic configurations.
        self.model_name = model
        self.vanilla = vanilla
        self.batch_size = batch_size
        self.target = tvm.target.Target(target, host="llvm -mtriple=aarch64-linux-gnu -mattr=+neon -num-cores=6")
        if self.target.attrs.get("mtriple", None) == "aarch64-linux-gnu":
            self.alloc_repeat = 3
        else:
            self.alloc_repeat = 1
        if rpc_host and rpc_port and rpc_key:
            self.rpc_config = ms.runner.RPCConfig(
                tracker_host=rpc_host,
                tracker_port=rpc_port,
                tracker_key=rpc_key,
                session_timeout_sec=rpc_timeout_sec,
            )
            self.workers = self.rpc_config.count_num_servers(allow_missing=False)
        else:
            # check all rpc configs are None
            assert (
                (rpc_host is None) and (rpc_port is None) and (rpc_key is None)
            ), "Please set all 'rpc_host', 'rpc_port' and 'rpc_key' to use PRC server"
            self.rpc_config = None
            self.workers = 1
        self.specialized_dir_name = os.path.join(self.target.kind.name, get_specialized_model_name(self.model_name, self.batch_size, vanilla=self.vanilla))
        self.working_dir = os.path.join(working_dir, self.specialized_dir_name)
        os.makedirs(self.working_dir, exist_ok=True)
        self.num_measurement_repeats = num_measurement_repeats
        self.num_measurements = num_measurements

        # Load the model.
        self.templated_mod, self.all_mappings, self.input_shape = import_templated_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_relax"), self.model_name, self.batch_size, vanilla=self.vanilla)

        # Input shape for runtime.
        self.input_dtype = "float32"
        self.input_info = {"inp_0": self.input_shape}

    def get_templated_mod(self) -> IRModule:
        return self.templated_mod

    def get_all_mappings(self) -> List[Dict[str, int]]:
        return self.all_mappings

    def get_kernel_specific_tuner(self, kernels_dir: Optional[os.PathLike], show: bool = False) -> KernelSpecificTuner:
        """Construct kernels according to the directory generated by Sampler.realize(). If the directory is None, return the original model."""
        if kernels_dir is None:
            working_dir = self.working_dir
            relax_mod = substitute_kernels(self.templated_mod, None)
        else:
            working_dir = os.path.join(kernels_dir, "perf", self.specialized_dir_name)
            os.makedirs(working_dir, exist_ok=True)
            if os.path.exists(os.path.join(kernels_dir, "kernel_scheduler_dir")):
                kernels_file = os.path.join(kernels_dir, "kernel_scheduler_dir", "kernels_tvm.py")
            else:
                kernels_file = os.path.join(kernels_dir, "kernels_tvm.py")
            global _TOTAL_KERNELS_SUBSTITUTED
            spec = importlib.util.spec_from_file_location(f"kas_tvm_kernels_mod_{_TOTAL_KERNELS_SUBSTITUTED}", kernels_file)
            _TOTAL_KERNELS_SUBSTITUTED += 1
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            kernel_builder = construct_kernel_builder(self.all_mappings, mod.weights, mod.build)
            relax_mod = substitute_kernels(self.templated_mod, kernel_builder)
        if show:
            print("After substitution of kernels:")
            relax_mod.show()
        return KernelSpecificTuner(self, working_dir, relax_mod)

    def _get_runner(self) -> ms.Runner:
        runner_config = {
            "evaluator_config": ms.runner.EvaluatorConfig(
                number=3,
                repeat=1,
                min_repeat_ms=100,
                enable_cpu_cache_flush=False,
            ),
            "alloc_repeat": self.alloc_repeat,
        }
        if self.rpc_config:
            runner = ms.runner.RPCRunner(
                rpc_config=self.rpc_config, max_workers=self.workers, **runner_config
            )
        else:
            runner = ms.runner.LocalRunner(**runner_config)
        return runner

    def _get_input_data(self) -> Dict[str, runtime.NDArray]:
        input_data: Dict[str, runtime.NDArray] = {}
        for input_name, input_shape in self.input_info.items():
            if self.input_dtype.startswith("float"):
                input_data[input_name] = np.random.uniform(size=input_shape).astype(self.input_dtype)
            else:
                input_data[input_name] = np.random.randint(
                    low=0, high=10000, size=input_shape, dtype=self.input_dtype
                )
        return input_data

    def _perform_measurement(
        self, rt_mod: runtime.Module, device: runtime.ndarray.Device, input_data: Dict[str, runtime.NDArray]
    ) -> runtime.module.BenchmarkResult:
        vm = relax.VirtualMachine(rt_mod, device=device)
        vm.save_function("main", "measure_func", **input_data, include_return=False)
        evaluator = vm.time_evaluator(
            func_name="measure_func",
            dev=device,
            repeat=self.num_measurement_repeats,
            number=self.num_measurements,
            min_repeat_ms=500,
        )
        return evaluator()

    def _measure(self, executable: relax.Executable) -> runtime.module.BenchmarkResult:
        input_data = self._get_input_data()
        if self.rpc_config:
            result = run_module_via_rpc(
                rpc_config=self.rpc_config,
                lib=executable.mod,
                dev_type=self.target.kind.name,
                args=input_data,
                continuation=self._perform_measurement,
            )
        else:
            dev = tvm.device(self.target.kind.name)
            result = self._perform_measurement(executable.mod, dev, input_data)
        return result

def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model",
        type=str,
        default="resnet34layers",
    )
    args.add_argument(
        "--vanilla",
        type=lambda x: bool(strtobool(x)),
        default=None,
    )
    args.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    args.add_argument(
        "--target",
        type=str,
        default="llvm -mtriple=aarch64-linux-gnu -mattr=+neon -num-cores=6",
    )
    args.add_argument(
        "--kernels-dir",
        type=str,
        default=None,
    )
    args.add_argument(
        "--num-trials",
        type=int,
        default=4000,
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
        default="./measurements",
    )
    return args.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    vanilla = args.vanilla if args.vanilla is not None else args.kernels_dir is None
    tuner = MetaScheduleTuner(
        model=args.model,
        vanilla=vanilla,
        batch_size=args.batch_size,
        target=args.target,
        rpc_host=args.rpc_host,
        rpc_port=args.rpc_port,
        rpc_key=args.rpc_key,
        working_dir=args.working_dir,
    )
    kernels_tuner = tuner.get_kernel_specific_tuner(args.kernels_dir, show=True)
    kernels_tuner.optimize_model_before_tuning(show=True)
    kernels_tuner.tune(args.num_trials)
    executable = kernels_tuner.build(show=True)
    results = kernels_tuner.measure(executable)
    print("results:", results)
