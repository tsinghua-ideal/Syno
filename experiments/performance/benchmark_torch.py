import argparse
from importlib.util import spec_from_file_location, module_from_spec
import os
from typing import Tuple, Type

import torch
torch.set_float32_matmul_precision('high')

import triton

from common import get_specialized_model_name


def import_torch_model(
    save_path: str,
) -> Tuple[Type[torch.nn.Module], Tuple[int, ...]]:
    save_path = os.path.join(save_path, "module.py")
    spec = spec_from_file_location("ExportedModule", save_path)
    assert spec is not None, f"Failed to load module from {save_path}"
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ExportedModel, mod.INPUT_SHAPE


def do_profile(fn, warmup=100, rep=500, grad_to_none=None):
    outputs = fn()
    torch.cuda.synchronize()

    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=outputs.device)

    # Estimate the runtime of the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = min(6, max(1, int(rep / estimate_ms)))
    # Warm-up
    for _ in range(n_warmup):
        fn()
    torch.cuda.cudart().cudaProfilerStart()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        torch.cuda.nvtx.range_push("iteration")
        fn()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def run_benchmark(
    Model: Type[torch.nn.Module],
    input_shape: Tuple[int, ...],
    device: torch.device,
    mode: str = "none",
    profile: bool = False,
):
    model = Model().to(device).eval()
    inputs = torch.randn(input_shape, device=device, requires_grad=False)
    # model = torch.cuda.make_graphed_callables(model, (inputs,))
    with torch.no_grad():
        if mode != "none":
            model = torch.compile(
                model,
                backend="inductor",
                mode=mode,
                dynamic=False,
                fullgraph=True,
            )
        print("Compiling model...")
        # Warmup
        model(inputs)
        print("Running benchmark...")
        # Use triton benchmark if using CUDA
        if device.type == "cuda":
            if profile:
                return do_profile(
                    lambda: model(inputs),
                )
            else:
                return triton.testing.do_bench(
                    lambda: model(inputs),
                    warmup=100,
                    rep=500,
                    return_mode="median",
                )
        else:
            import torch.utils.benchmark as benchmark
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
    parser.add_argument("--mode", type=str, default="none")
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    specialized_model_name = get_specialized_model_name(
        args.model, args.batch_size, vanilla=True
    )
    model_path = os.path.join("model_torch", specialized_model_name)
    Model, input_shape = import_torch_model(model_path)
    device = torch.device(args.device)
    print(run_benchmark(Model, input_shape, device, mode=args.mode, profile=args.profile))
    print("Done.")
