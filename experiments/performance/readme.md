# Ansor for KAS

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

to start the client.

## How to run

```bash
python MetaScheduleTuner.py --batch-size 1 --model torchvision/resnet18 --kernels-dir "/path/to/kernels/dir"
```

If `--kernels-dir` is not specified, the original network will be benchmarked.

### How to use GPU

Add option `--target "cuda -arch=sm_87 -max_threads_per_block=128 -max_num_threads=1024 -thread_warp_size=32 -max_shared_memory_per_block=49152 -registers_per_block=65536"`.

This does not work for the time being.
