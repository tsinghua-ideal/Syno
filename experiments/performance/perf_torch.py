import argparse
import filelock
import math
import os
import statistics
import sys

import torch

from common import PRESET_WORKING_DIR, get_specialized_model_name
from import_kas import get_model

_BENCHMARK_TIME_WARMUP = 100
_BENCHMARK_TIME_REPEAT = 500


def _quantile(a, q):
    n = len(a)
    a = sorted(a)

    def get_quantile(q):
        if not (0 <= q <= 1):
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = q * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * a[lower] + t * a[upper]

    return [get_quantile(q) for q in q]


def _summarize_statistics(times, quantiles, return_mode):
    if quantiles is not None:
        ret = _quantile(times, quantiles)
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times
    elif return_mode == "min":
        return min(times)
    elif return_mode == "max":
        return max(times)
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)


def do_benchmark_cuda(fn, warmup=_BENCHMARK_TIME_WARMUP, rep=_BENCHMARK_TIME_REPEAT, quantiles=None, return_mode="mean"):
    assert return_mode in ["min", "max", "mean", "median", "all"]

    fn()
    torch.cuda.synchronize()

    # Estimate the runtime of the function
    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)
    torch.cuda._sleep(10_000_000)
    start_event.record()
    for _ in range(5):
        torch.cuda._sleep(1_000_000)
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(6, int(rep / estimate_ms))
    start_event = [torch.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    torch.cuda._sleep(10_000_000)
    for i in range(n_repeat):
        # record time of `fn`
        torch.cuda._sleep(1_000_000)
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, return_mode)


_TORCH_TUNING_MODE="max-autotune"
_TORCH_EAGER_MODE="eager"


def run_benchmark(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int, int],
    device: torch.device,
    mode: str,
):
    model = model.to(device).eval()
    inputs = torch.randn(input_shape, device=device, requires_grad=False)
    with torch.no_grad():
        if mode != _TORCH_EAGER_MODE:
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
        for _ in range(5):
            run_model(inputs)
        with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
            print("Running benchmark...")
            # Use triton benchmark if using CUDA
            if device.type == "cuda":
                return do_benchmark_cuda(
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
    parser.add_argument("--mode", type=str, default=_TORCH_TUNING_MODE)
    parser.add_argument("--disable-tf32", action="store_true")
    return parser.parse_args()


def get_benchmark_output_path(model_name: str, target_type: str, mode: str, kernels_dir: str | None, batch_size: int = 1) -> str:
    specialized_name = os.path.join(
        f"inductor-{mode}-{target_type}",
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
        mode=args.mode,
        kernels_dir=args.result_dir,
        batch_size=args.batch_size,
    )

    os.makedirs(os.path.dirname(benchmark_output), exist_ok=True)
    benchmark_lock = f"{benchmark_output}.lock"
    with filelock.FileLock(benchmark_lock):
        if os.path.exists(benchmark_output):
            print(f"Benchmark output {benchmark_output} already exists, skipping.")
            return
        # Create the file to prevent other processes from running the same benchmark
        with open(benchmark_output, "w"):
            pass

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

    try:
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
    except:
        # Remove the file if an exception occurs
        os.remove(benchmark_output)
        raise

    print(f"Writing benchmark output to {benchmark_output}")
    with open(benchmark_output, "w") as f:
        f.write(f"{benchmark_time}\n")
    print("Done.")


if __name__ == "__main__":
    main()
