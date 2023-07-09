import logging
import KAS
import torch
import torch.nn as nn
import os

size_N = 100
size_C_in = 3
size_C_out = 16
size_H = 128
size_W = 128
size_K = 5
size_input = size_N * size_C_in * size_H * size_W
size_output = size_N * size_C_out * size_H * size_W

device = torch.device("cuda:0")

loader = KAS.Bindings.Loader(KAS.Bindings.LoaderArgs(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../build/tests/Semantics/kernel_conv2d/kernels.so"),
    "conv2d",
    True,
    2,
    1,
    [0]
))

# Run CTests before this to generate the kernel
pack = KAS.KernelPack(
    identifier="conv2d",
    loader=loader,
    index=0,
    unpadded_inputs_shapes=[[size_N, size_C_in, size_H, size_W], [size_C_out, size_C_in, size_K, size_K]],
    padded_inputs_shapes=[[size_N, size_C_in, size_H, size_W], [size_C_out, size_C_in, size_K, size_K]],
    unpadded_output_shape=[size_N, size_C_out, size_H, size_W],
    padded_output_shape=[size_N, size_C_out, size_H, size_W],
    device=device)
kas_conv = KAS.Placeholder({})
kas_conv.reload(pack)

pack.weights = nn.ParameterList([torch.randn([size_C_out, size_C_in, size_K, size_K], device=device)])

import torch.utils.benchmark as benchmark

from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("kernel_backward"):
        t_in = torch.randn([size_N, size_C_in, size_H, size_W], requires_grad=True, device=device)
        torch.sum(kas_conv(t_in)).backward()
        torch.cuda.synchronize()
# wait
prof.export_chrome_trace("trace.json")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
