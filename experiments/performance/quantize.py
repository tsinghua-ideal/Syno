import argparse
import numpy as np
import os
import sys
import torch
from torch import fx
import torchvision
from typing import Callable, cast, Dict, Optional, Tuple, TypeVar, Union

import tvm
from tvm import relax, relay, runtime, transform
import tvm.relay.testing
from tvm.relax.testing import relay_translator
from tvm.relax.frontend.torch import from_fx
from tvm.contrib import graph_executor
from tvm.contrib.tar import tar
from tvm.ir.module import IRModule
from tvm import meta_schedule as ms

from MetaScheduleTuner import (
    BaseMetaScheduleTuner,
    KernelSpecificTuner,
    PRESET_TARGET,
    PRESET_RPC,
    PRESET_RPC_HOST,
    PRESET_RPC_PORT,
    PRESET_RPC_KEY,
    PRESET_RPC_CONFIG,
    PRESET_WORKING_DIR,
    parse_target,
    parse_rpc_config,
)


def quantize_model(model, inp):
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
    torch.quantization.prepare(model, inplace=True)
    # Dummy calibration
    model(inp)
    torch.quantization.convert(model, inplace=True)


class QuantizedResnet18Tuner(BaseMetaScheduleTuner):
    def __init__(
        self,
        target: tvm.target.Target = PRESET_TARGET,
        rpc_config: Optional[ms.runner.RPCConfig] = PRESET_RPC_CONFIG,
    ) -> None:
        super().__init__(target, rpc_config)

    @property
    def input_dtype(self) -> str:
        return "float32"

    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (1, 3, 224, 224)

    @property
    def model_name(self) -> str:
        return "resnet18_quantized"


def tune_quantized_resnet18_relax(
    target: tvm.target.Target,
    rpc_config: Optional[ms.runner.RPCConfig],
    working_dir: str,
    num_trials: int,
) -> None:
    working_dir = os.path.join(working_dir, target.kind.name, "qresnet18")

    base_tuner = QuantizedResnet18Tuner(target, rpc_config)

    model = torchvision.models.quantization.resnet18(pretrained=True).eval()
    pt_inp = torch.rand(base_tuner.input_shape)
    quantize_model(model, pt_inp)
    script_model = torch.jit.trace(model, pt_inp).eval()
    relay_mod, params = relay.frontend.from_pytorch(
        script_model, list(base_tuner._get_input_shape().items())
    )
    with transform.PassContext(opt_level=3):
        main_func = relay_mod["main"]
        bind_main_func = relay.build_module.bind_params_by_name(main_func, params)
        relay_mod = IRModule.from_expr(bind_main_func)
        relay_mod = relay.transform.SimplifyInference()(relay_mod)
        relay_mod = relay.transform.FoldConstant()(relay_mod)
        relay_mod = relay.transform.FoldScaleAxis()(relay_mod)
        relay_mod = relay.transform.CanonicalizeOps()(relay_mod)
        relay_mod = relay.transform.AlterOpLayout()(relay_mod)
        relay_mod = relay.transform.FoldConstant()(relay_mod)
        relax_mod = relay_translator.from_relay(relay_mod["main"], target=target)
    tuner = KernelSpecificTuner(base_tuner, working_dir, relax_mod)
    tuner.tune_e2e(num_trials)


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--target",
        type=str,
        default=None,
    )
    args.add_argument(
        "--target-host",
        type=str,
        default=None,
        help="If target is GPU, host code is required to launch the kernel on the GPU, so a host-side target is also required.",
    )
    args.add_argument(
        "--target-preset",
        type=str,
        default=None,
        help="Use a preset configuration for --target and --target-host.",
    )
    args.add_argument(
        "--num-trials",
        type=int,
        default=10000,
    )
    args.add_argument(
        "--rpc",
        action="store_true",
        default=PRESET_RPC,
        help="Use RPC. Please further set --rpc-host, --rpc-port, and --rpc-key.",
    )
    args.add_argument(
        "--rpc-host",
        type=str,
        default=PRESET_RPC_HOST,
    )
    args.add_argument(
        "--rpc-port",
        type=int,
        default=PRESET_RPC_PORT,
    )
    args.add_argument(
        "--rpc-key",
        type=str,
        default=PRESET_RPC_KEY,
    )
    args.add_argument(
        "--working-dir",
        type=str,
        default=PRESET_WORKING_DIR,
    )
    return args.parse_args()


def main(args: argparse.Namespace) -> int:
    target = parse_target(args.target, args.target_host, args.target_preset)
    rpc_config = parse_rpc_config(args.rpc, args.rpc_host, args.rpc_port, args.rpc_key)
    tune_quantized_resnet18_relax(
        target=target,
        rpc_config=rpc_config,
        working_dir=args.working_dir,
        num_trials=args.num_trials,
    )
    return 0


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main(args))
