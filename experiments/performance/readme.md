# MetaSchedule for KAS

## Exporting the Network

```bash
python export_relax.py --batch-size 1 --model torchvision/resnet18
```

## How to setup

On the host, run

```bash
python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
```

to start the tracker.

On your target device, which in our case is NVIDIA Jetson Orin Nano, run

```bash
python -m tvm.exec.rpc_server --tracker=[HOST_IP]:9190 --key=jetson-orin-nano
```

for example, if you are running the tracker on the host machine, then it is

```bash
python -m tvm.exec.rpc_server --tracker=127.0.0.1:9190 --key=jetson-orin-nano
```

or, on A100 GPU, run

```bash
CUDA_VISIBLE_DEVICES=0 python -m tvm.exec.rpc_server --tracker=127.0.0.1:9190 --key=a100
```

to start the client.

## How to run

```bash
python MetaScheduleTuner.py --batch-size 1 --model torchvision/resnet18 --target-preset "jetson_orin_nano-cpu" --kernels-dir "/path/to/kernels/dir"
```

If you need a custom target, you can specify that by `--target <your-target> --target-host <your-target-host>`. GPU targets must specify `--target-host` because host-side code generation requires that.

If `--kernels-dir` is not specified, the original network will be benchmarked, but still the placeholders will be substituted, which may alter the original convolution sizes or strides. To benchmark the original network, use `--vanilla`.

### How to make a few trials instead of the full run

Use `--num-trials 60 --max-trials-per-task 2 --num-trials-per-iter 2` for a quick proof-of-concept run. Moreover, you can specify the working directory by `--working-dir <your-testing-directory>`.

### Presets

RPC run, NVIDIA Jetson Orin Nano CPU:
```
python MetaScheduleTuner.py --batch-size 1 --model torchvision/resnet18 --target-preset "jetson_orin_nano-cpu"
```

RPC run, NVIDIA Jetson Orin Nano GPU:
```
python MetaScheduleTuner.py --batch-size 1 --model torchvision/resnet18 --target-preset "jetson_orin_nano-gpu"
```

Local run, x86_64 16-core CPU:
```
python MetaScheduleTuner.py --batch-size 1 --model torchvision/resnet18 --target-preset "x86_64-16_core_cpu" --no-rpc
```

Local run, RTX 3060 Laptop on x86_64 16-core CPU host:
```
python MetaScheduleTuner.py --batch-size 1 --model torchvision/resnet18 --target-preset "rtx_3060_laptop_gpu" --no-rpc
```

Recommended number of trials:

| Network | Tasks | Trials |
| ResNet-18 | 18 | 10000 |
| ResNet-34 | 18 | 10000 |
| ResNet-34 Layers | 1 | 4000 |
| ResNext-29 | 25 | 15000 |
| EfficientNet-v2 | 47 | 30000 |
| DenseNet-121 | 134 | 50000 |

These numbers suffice for convergence.

# PyTorch Benchmark

## Exporting the Network

```bash
python export_torch.py --batch-size 1 --model torchvision/resnet18
```

## How to run

First `rsync` the exported network to the target device. Then

```bash
python benchmark_torch.py --batch-size 1 --model torchvision/resnet18 --device cuda
```

If you need to profile the network, use `--profile`.

```bash
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu --capture-range=cudaProfilerApi --cudabacktrace=true -x true -o profile_resnet18 python benchmark_torch.py --batch-size 1 --model torchvision/resnet18 --device cuda --profile
```
