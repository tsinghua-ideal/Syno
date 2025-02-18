import argparse
import os
import sys

import torch

import triton

from common import PRESET_WORKING_DIR, get_specialized_model_name
from import_kas import get_model

_BENCHMARK_TIME_WARMUP = 100
_BENCHMARK_TIME_REPEAT = 500


def run_benchmark(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int, int],
    device: torch.device,
    mode: str = "none",
):
    model = model.to(device).eval()
    inputs = torch.randn(input_shape, device=device, requires_grad=False)
    with torch.no_grad():
        if mode != "none":
            run_model = torch.compile(
                model,
                backend="inductor",
                mode=mode,
                dynamic=False,
                fullgraph=True,
            )
        else:
            run_model = model
        print("Compiling model...")
        # Warmup
        run_model(inputs)
        print("Running benchmark...")
        # Use triton benchmark if using CUDA
        if device.type == "cuda":
            return triton.testing.do_bench(
                lambda: run_model(inputs),
                warmup=_BENCHMARK_TIME_WARMUP,
                rep=_BENCHMARK_TIME_REPEAT,
                return_mode="mean",
            )
        else:
            import torch.utils.benchmark as benchmark
            timer = benchmark.Timer(
                stmt="run_model(inputs)",
                globals={"run_model": run_model, "inputs": inputs},
                label="torchscript",
            )
            return timer.blocked_autorange(min_run_time=1).mean


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--result-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument(
        "--input-size",
        default=(3, 224, 224),
        nargs=3,
        type=int,
        metavar="N N N",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", type=str, default="none")
    parser.add_argument("--disable-tf32", action="store_true")
    return parser.parse_args()


def get_benchmark_output_path(model_name: str, target_type: str, kernels_dir: str | None, batch_size: int = 1) -> str:
    specialized_name = os.path.join(
        f"inductor-{target_type}",
        get_specialized_model_name(
            model_name,
            batch_size=batch_size,
            vanilla=kernels_dir is None,
        ),
    )
    specialized_file = f"{specialized_name}.txt"
    if kernels_dir is None:
        output_path = os.path.join(PRESET_WORKING_DIR, specialized_file)
    else:
        output_path = os.path.join(kernels_dir, "perf", specialized_file)
    return output_path


def main():
    args = _parse_args()

    benchmark_output = get_benchmark_output_path(
        model_name=args.model,
        target_type=args.device,
        kernels_dir=args.result_dir,
        batch_size=args.batch_size,
    )

    if os.path.exists(benchmark_output):
        print(f"Benchmark output {benchmark_output} already exists, skipping.")
        return

    # Disable all TorchInductor caching, so that different experiments are not affected by each other.
    import torch._inductor.config
    torch._inductor.config.force_disable_caches = True

    # Enable benchmarking mode
    torch.backends.cudnn.benchmark = True

    # Disable TF32 if requested
    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, input_shape = get_model(
        model_name=args.model,
        result_dir=args.result_dir,
        batch_size=args.batch_size,
        input_size=args.input_size,
        num_classes=args.num_classes,
    )
    device = torch.device(args.device)
    benchmark_time = run_benchmark(model, input_shape, device, mode=args.mode)
    print(f"Benchmark time: {benchmark_time:.3f} ms")
    print(f"Writing benchmark output to {benchmark_output}")
    os.makedirs(os.path.dirname(benchmark_output), exist_ok=True)
    with open(benchmark_output, "w") as f:
        f.write(f"{benchmark_time}\n")
    print("Done.")


if __name__ == "__main__":
    main()
