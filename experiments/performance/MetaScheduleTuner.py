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
from tvm import relay, relax, runtime, te, transform
from tvm.ir.module import IRModule
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.relax.testing import relay_translator
from tvm.target.target import Target


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model",
        type=str,
        default="ConvNet",
    )
    args.add_argument(
        "--input-shape",
        type=str,
        default="[1, 3, 32, 32]",
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
        "--cache-dir",
        type=str,
        default="./cached_networks",
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
    parsed.input_shape = json.loads(parsed.input_shape)
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
        # relay_mod = relay.transform.SimplifyInference()(relay_mod)
        # relay_mod = relay.transform.FoldConstant()(relay_mod)
        # relay_mod = relay.transform.FoldScaleAxis()(relay_mod)
        # relay_mod = relay.transform.CanonicalizeOps()(relay_mod)
        # relay_mod = relay.transform.AlterOpLayout()(relay_mod)
        # relay_mod = relay.transform.FoldConstant()(relay_mod)

        relax_mod = relax.transform.LegalizeOps()(relax_mod)
        # relax_mod = relax.transform.DecomposeOpsForInference()(relax_mod)
        # relax_mod = relax.transform.FoldConstant()(relax_mod)

        # relax_mod = relax.transform.AnnotateTIROpPattern()(relax_mod)
        # relax_mod = relax.transform.FuseOps()(relax_mod)
        # relax_mod = relax.transform.FuseTIR()(relax_mod)
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

def _shape_expr_to_tuple(shape_expr: relax.ShapeExpr) -> tuple:
    return tuple(v.value for v in shape_expr.values)

def substitute_kernels(relax_mod: IRModule) -> IRModule:
    @relax.expr_functor.mutator
    class KernelReplacer(relax.PyExprMutator):
        def __init__(self, mod: IRModule) -> None:
            super().__init__()
            self.mod_ = mod

        def visit_call_(self, call):
            call = self.visit_expr_post_order(call)

            # This is a hack. Since we cannot pass information to Relax, we rewrite the layer to be replaced to this form:
            #   layer(x) -> exp(layer(x - kernel_id))
            # In this way we can know which kernels we should replace and the kernel ids.

            exp_op = tvm.ir.Op.get("relax.exp")
            add_op = tvm.ir.Op.get("relax.add")
            conv2d_op = tvm.ir.Op.get("relax.nn.conv2d")
            subtract_op = tvm.ir.Op.get("relax.subtract")

            # First match the outer exp.
            if call.op != exp_op:
                return call

            original_output = self.lookup_binding(call.args[0])

            logging.info(f"Processing {original_output}...")

            # Relax conv2d does not support bias, and bias is implemented by another add op.
            if original_output.op == add_op:
                has_bias = True
                conv_call = self.lookup_binding(original_output.args[0])
                conv_bias = original_output.args[1]
            elif original_output.op == conv2d_op:
                has_bias = False
                conv_call = original_output
                conv_bias = None
            else:
                assert False, f"Unknown op {original_output.op} when handling a marked kernel!"

            # x - kernel_id
            marked_input = self.lookup_binding(conv_call.args[0])
            assert isinstance(marked_input, relax.Call) and marked_input.op == subtract_op, f"Marked input is illegal: {marked_input}"

            # x
            data = marked_input.args[0]
            logging.info(f"data = {data}")
            conv_weight = conv_call.args[1]
            logging.info(f"conv_weight = {conv_weight}")
            if has_bias:
                logging.info(f"conv_bias = {conv_bias}")

            kernel_id = marked_input.args[1]
            assert isinstance(kernel_id, relax.Constant)
            kernel_id = int(kernel_id.data.numpy())
            logging.info(f"kernel_id = {kernel_id}")

            sinfo_input = data.struct_info
            logging.info(f"sinfo_input = {sinfo_input.shape.values}")
            sinfo_output = original_output.struct_info
            logging.info(f"sinfo_output = {sinfo_output.shape.values}")

            # Restore the kernel. TODO: replace.
            new_output = self.builder_.emit(
                relax.op.nn.conv2d(
                    data,
                    conv_weight,
                    strides=conv_call.attrs.strides,
                    padding=conv_call.attrs.padding,
                    dilation=conv_call.attrs.dilation,
                    groups=conv_call.attrs.groups,
                    data_layout=conv_call.attrs.data_layout,
                    kernel_layout=conv_call.attrs.kernel_layout,
                    out_dtype=conv_call.attrs.out_dtype,
                )
            )
            if has_bias:
                new_output = self.builder_.emit(relax.op.add(new_output, conv_bias))
            logging.info(f"Rewritten.")

            # Proof of concept.
            interesting_bias = relax.const(np.zeros(_shape_expr_to_tuple(sinfo_output.shape), "float32"))
            def te_func(X: te.Tensor, Y: te.Tensor):
                return te.compute(sinfo_output.shape.values, lambda *i: X[i] + Y[i])
            biased = self.builder_.emit_te(te_func, new_output, interesting_bias)

            return biased

        def transform(self) -> IRModule:
            for global_var, func in self.mod_.functions.items():
                if not isinstance(func, relax.Function):
                    continue
                updated_func = self.visit_expr(func)
                self.builder_.update_func(global_var, updated_func)

            return self.builder_.get()

    @tvm.ir.transform.module_pass(opt_level=0, name="KASReplaceKernels")
    class ReplaceKernelsPass:
        """The wrapper for the LowerTensorIR pass."""
        def transform_module(self, mod, ctx):
            return KernelReplacer(mod).transform()

    relax_mod = ReplaceKernelsPass()(relax_mod)
    relax_mod = relax.transform.DeadCodeElimination(["main"])(relax_mod) # To remove markers.
    relax_mod = relax.transform.CanonicalizeBindings()(relax_mod)
    return relax_mod

def main():
    input_name = "inp_0"
    input_dtype = "float32"
    input_info = {input_name: ARGS.input_shape}
    input_data = {}
    for input_name, input_shape in input_info.items():
        logging.info(f"  input_name: {input_name}")
        logging.info(f"  input_shape: {input_shape}")
        logging.info(f"  input_dtype: {input_dtype}")

    # import the Relax model
    py_mod_name = f"model_relax.{ARGS.model}"
    assert py_mod_name not in sys.modules
    spec = importlib.util.spec_from_file_location(py_mod_name, f"model_relax/{ARGS.model}.py")
    py_mod = importlib.util.module_from_spec(spec)
    sys.modules[py_mod_name] = py_mod
    spec.loader.exec_module(py_mod)
    relax_mod = py_mod.Module
    assert isinstance(relax_mod, tvm.IRModule)

    relax_mod = substitute_kernels(relax_mod)
    relax_mod.show()
    # input()

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
        writer.writerow(["input_shape", ARGS.input_shape])
        writer.writerow(["target", ARGS.target])
        writer.writerow(["num_measurement_repeats", ARGS.num_measurement_repeats])
        for res in result.results:
            writer.writerow([str(res)])


if __name__ == "__main__":
    main()
