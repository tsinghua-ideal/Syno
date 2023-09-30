# Ansor for KAS

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

to start the client.

## How to run

Follow the example in `tests/`.
