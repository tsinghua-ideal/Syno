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
import datetime
import os
import sys
import csv
import json
import argparse
import logging
from typing import Dict, Optional
import numpy as np  # type: ignore
import importlib

import tvm
from tvm import relax, runtime, te, transform
from tvm.ir.module import IRModule
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.target.target import Target

from Model import import_templated_model, substitute_kernels


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model",
        type=str,
        default="ConvNet",
    )
    args.add_argument(
        "--target",
        type=str,
        default="llvm -num-cores 16",
    )
    args.add_argument(
        "--num-trials",
        type=int,
        default=10,
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
        "--work-dir",
        type=str,
        default="./measurements",
    )
    args.add_argument(
        "--rpc-timeout-sec",
        type=int,
        default=180,
    )
    args.add_argument("--num-measurement-repeats", type=int, default=3)
    args.add_argument("--num-measurements", type=int, default=2)
    args.add_argument("--results-file", type=str, default="./results.csv")
    parsed = args.parse_args()
    parsed.target = tvm.target.Target(parsed.target)
    if parsed.target.attrs.get("mtriple", None) == "aarch64-linux-gnu":
        parsed.alloc_repeat = 3
    else:
        parsed.alloc_repeat = 1
    if parsed.rpc_host and parsed.rpc_port and parsed.rpc_key:
        parsed.rpc_config = ms.runner.RPCConfig(
            tracker_host=parsed.rpc_host,
            tracker_port=parsed.rpc_port,
            tracker_key=parsed.rpc_key,
            session_timeout_sec=parsed.rpc_timeout_sec,
        )
        parsed.workers = parsed.rpc_config.count_num_servers(allow_missing=False)
    else:
        # check all rpc configs are None
        assert (
            (parsed.rpc_host is None) and (parsed.rpc_port is None) and (parsed.rpc_key is None)
        ), "Please set all 'rpc_host', 'rpc_port' and 'rpc_key' to use PRC server"
        parsed.rpc_config = None
        parsed.workers = 1
    return parsed

logging.basicConfig(level=logging.DEBUG)
ARGS = _parse_args()

def apply_opt_before_tuning(relax_mod: IRModule, target: Target) -> IRModule:
    with transform.PassContext(opt_level=3):
        relax_mod = relax.transform.DecomposeOpsForInference()(relax_mod)
        relax_mod = relax.transform.LegalizeOps(enable_warning=True)(relax_mod)
        relax_mod = relax.transform.AnnotateTIROpPattern()(relax_mod)
        relax_mod = relax.transform.FoldConstant()(relax_mod)
        relax_mod = relax.transform.FuseOps()(relax_mod)
        relax_mod = relax.transform.FuseTIR()(relax_mod)
    return relax_mod

def f_measurement(
    rt_mod: runtime.Module, device: runtime.ndarray.Device, input_data: Dict[str, runtime.NDArray]
):
    vm = relax.VirtualMachine(rt_mod, device=device)
    vm.save_function("main", "measure_func", **input_data, include_return=False)
    evaluator = vm.time_evaluator(
        func_name="measure_func",
        dev=device,
        repeat=ARGS.num_measurement_repeats,
        number=ARGS.num_measurements,
        min_repeat_ms=500,
    )
    return evaluator()

def get_runner():
    runner_config = {
        "evaluator_config": ms.runner.EvaluatorConfig(
            number=3,
            repeat=1,
            min_repeat_ms=100,
            enable_cpu_cache_flush=False,
        ),
        "alloc_repeat": ARGS.alloc_repeat,
    }
    if ARGS.rpc_config:
        runner = ms.runner.RPCRunner(
            rpc_config=ARGS.rpc_config, max_workers=ARGS.workers, **runner_config
        )
    else:
        runner = ms.runner.LocalRunner(**runner_config)

    return runner

def main():
    # import the Relax model
    relax_mod, loaded_input_shape = import_templated_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_relax"), ARGS.model)

    relax_mod = substitute_kernels(relax_mod, None)
    logging.info("After substitute_kernels:")
    relax_mod.show()

    relax_mod = apply_opt_before_tuning(relax_mod, ARGS.target)

    db = ms.relax_integration.tune_relax(
        mod=relax_mod,
        target=ARGS.target,
        params=None,
        num_trials_per_iter=64,
        max_trials_per_task=ARGS.num_trials,
        max_trials_global=ARGS.num_trials,
        runner=get_runner(),
        work_dir=ARGS.work_dir,
    )
    executable = ms.relax_integration.compile_relax(
        db,
        mod=relax_mod,
        target=ARGS.target,
        params=None,
    )

    input_name = "inp_0"
    input_dtype = "float32"
    input_info = {input_name: loaded_input_shape}
    input_data = {}
    for input_name, input_shape in input_info.items():
        logging.info(f"  input_name: {input_name}")
        logging.info(f"  input_shape: {input_shape}")
        logging.info(f"  input_dtype: {input_dtype}")

    for input_name, input_shape in input_info.items():
        if input_dtype.startswith("float"):
            input_data[input_name] = np.random.uniform(size=input_shape).astype(input_dtype)
        else:
            input_data[input_name] = np.random.randint(
                low=0, high=10000, size=input_shape, dtype=input_dtype
            )

    # for documentation purposes
    start_time = datetime.datetime.now()

    if ARGS.rpc_config:
        result = run_module_via_rpc(
            rpc_config=ARGS.rpc_config,
            lib=executable.mod,
            dev_type=ARGS.target.kind.name,
            args=input_data,
            continuation=f_measurement,
        )
    else:
        dev = tvm.device(ARGS.target.kind.name)
        result = f_measurement(executable.mod, dev, input_data)

    logging.info(result)

    if not ARGS.results_file:
        return

    out_path = os.path.abspath(os.path.expanduser(ARGS.results_file))
    with open(out_path, "w") as out_file:
        writer = csv.writer(out_file)
        # write experiment parameters at the top as a record
        writer.writerow(["start", str(start_time)])
        writer.writerow(["model", ARGS.model])
        writer.writerow(["input_shape", loaded_input_shape])
        writer.writerow(["target", ARGS.target])
        writer.writerow(["num_measurement_repeats", ARGS.num_measurement_repeats])
        for res in result.results:
            writer.writerow([str(res)])


if __name__ == "__main__":
    main()
