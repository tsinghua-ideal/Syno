import numpy as np
import os
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

from MetaScheduleTuner import BaseMetaScheduleTuner, KernelSpecificTuner, PRESET_TARGET, PRESET_RPC_CONFIG, PRESET_WORKING_DIR, parse_target


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

def tune_quantized_resnet18_relay() -> None:
    target = parse_target(None, None, "jetson_orin_nano-gpu")

    working_dir = os.path.join("./perf", target.kind.name, "qresnet18")
    os.makedirs(working_dir, exist_ok=True)
    lib_path = os.path.join(working_dir, "kernels_tvm_tuned.tar")

    base_tuner = QuantizedResnet18Tuner(target)
    if not os.path.exists(lib_path):
        model = torchvision.models.quantization.resnet18(pretrained=True).eval()
        pt_inp = torch.rand(base_tuner.input_shape)
        quantize_model(model, pt_inp)
        script_model = torch.jit.trace(model, pt_inp).eval()
        tvm_model, params = relay.frontend.from_pytorch(script_model, list(base_tuner._get_input_shape().items()))
        tvm_model.show()

        database = ms.relay_integration.tune_relay(
            mod=tvm_model,
            target=target,
            params=params,
            work_dir=working_dir,
            max_trials_global=10000,
            num_trials_per_iter=64,
            runner=base_tuner.get_runner(),
        )
        lib = ms.relay_integration.compile_relay(database, tvm_model, target=target, params=params)
        lib.export_library(lib_path, tar)

    def measure_relay(
        rt_mod: runtime.Module,
        device: runtime.ndarray.Device,
        input_data: Dict[str, np.ndarray],
        num_measurement_repeats: int = 3,
        num_measurements: int = 2,
    ) -> runtime.module.BenchmarkResult:
        print(f"num_measurement_repeats: {num_measurement_repeats}, num_measurements: {num_measurements}")
        module = graph_executor.GraphModule(rt_mod["default"](device))
        module.set_input(**input_data)
        return module.benchmark(device, repeat=num_measurement_repeats, number=num_measurements)

    result = base_tuner.measure(lib_path, measure_relay)
    print(result)

def tune_quantized_resnet18_relax():
    target = parse_target(None, None, "jetson_orin_nano-cpu")
    working_dir = os.path.join("./perf", target.kind.name, "qresnet18")

    base_tuner = QuantizedResnet18Tuner(target)

    model = torchvision.models.quantization.resnet18(pretrained=True).eval()
    pt_inp = torch.rand(base_tuner.input_shape)
    quantize_model(model, pt_inp)
    script_model = torch.jit.trace(model, pt_inp).eval()
    relay_mod, params = relay.frontend.from_pytorch(script_model, list(base_tuner._get_input_shape().items()))
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
    tuner.tune_e2e(10000)

if __name__ == "__main__":
    tune_quantized_resnet18_relax()
