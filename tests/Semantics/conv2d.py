import KAS
import torch
import torch.nn as nn
import os

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
    unpadded_inputs_shapes=[[64, 3, 128, 128], [16, 3, 5, 5]],
    padded_inputs_shapes=[[64, 3, 128, 128], [16, 3, 5, 5]],
    unpadded_output_shape=[64, 16, 128, 128],
    padded_output_shape=[64, 16, 128, 128],
    device=device)
kas_conv = KAS.Placeholder({})
kas_conv.reload(pack)

pack.weights = nn.ParameterList([torch.randn([16, 3, 5, 5], device=device)])

t_in = torch.randn([64, 3, 128, 128], requires_grad=True, device=device)
torch.sum(kas_conv(t_in)).backward()
torch.cuda.synchronize()
