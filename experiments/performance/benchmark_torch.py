import argparse
from importlib.util import spec_from_file_location, module_from_spec
import os
from typing import Tuple, Type

import torch
import torch.utils.benchmark as benchmark
from torch.utils.benchmark.utils.common import Measurement

from .common import get_specialized_model_name


def import_torch_model(
    save_path: str,
) -> Tuple[Type[torch.nn.Module], Tuple[int, ...]]:
    save_path = os.path.join(save_path, "module.py")
    spec = spec_from_file_location("ExportedModule", save_path)
    assert spec is not None, f"Failed to load module from {save_path}"
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ExportedModel, mod.INPUT_SHAPE


def run_benchmark(
    Model: Type[torch.nn.Module],
    input_shape: Tuple[int, ...],
    device: torch.device,
) -> Measurement:
    model = torch.compile(
        Model().to(device),
        backend="inductor",
        mode="max-autotune",
        dynamic=False,
        fullgraph=True,
    )
    model.eval()
    inputs = torch.randn(input_shape, device=device)
    print("Compiling model...")
    # Warmup
    model(inputs)
    print("Running benchmark...")
    timer = benchmark.Timer(
        stmt="model(inputs)",
        globals={"model": model, "inputs": inputs},
        label="torchscript",
    )
    return timer.blocked_autorange(min_run_time=1)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    specialized_model_name = get_specialized_model_name(
        args.model, args.batch_size, vanilla=True
    )
    model_path = os.path.join("model_torch", specialized_model_name)
    Model, input_shape = import_torch_model(model_path)
    device = torch.device(args.device)
    print(run_benchmark(Model, input_shape, device))
    print("Done.")
