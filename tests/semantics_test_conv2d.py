import KAS
import torch
import torch.nn as nn
import os

device = torch.device("cuda:0")

# Run CTests before this to generate the kernel
pack = KAS.KernelPack("conv2d", os.path.join(os.path.dirname(os.path.realpath(__file__)), "../build/tests/kernel_conv2d"), "conv2d", [[64, 3, 128, 128], [16, 3, 5, 5]], [64, 16, 128, 128], device=device)
kas_conv = KAS.Placeholder({})
kas_conv.reload(pack)
torch_conv = nn.Conv2d(3, 16, (5, 5), bias=False, padding="same", padding_mode='zeros', device=device)

pack.weights = nn.ParameterList([torch.ones([16, 3, 5, 5], device=device)])
torch_conv.weight = nn.Parameter(torch.ones_like(torch_conv.weight, device=device))


assert torch.isclose(pack.weights[0], torch_conv.weight).all()
t_in = torch.ones([64, 3, 128, 128], device=device)

with torch.no_grad():
    k_out = kas_conv(t_in)
    t_out = torch_conv(t_in)
    print("forward_kas:", k_out.view(-1)[1500000:1500010])
    print("forward_torch:", t_out.view(-1)[1500000:1500010])
    print("forward is close:", torch.isclose(k_out, t_out).all())

t_in = torch.randn([64, 3, 128, 128], requires_grad=True, device=device)
torch.sum(kas_conv(t_in)).backward()
grad_kas = t_in.grad.detach()
pack.zero_grad(True)
t_in.grad = None
torch.sum(torch_conv(t_in)).backward()
grad_torch = t_in.grad.detach()
torch_conv.zero_grad(True)
t_in.grad = None
print("grad_kas:", grad_kas.view(-1)[1500000:1500010])
print("grad_torch:", grad_torch.view(-1)[1500000:1500010])
print("backward is close:", torch.isclose(grad_kas, grad_torch).all())


import torch.utils.benchmark as benchmark

t_in = torch.randn([64, 3, 128, 128], device=device)

kas_test_forward = '''
with torch.no_grad():
    kas_conv(t_in)
    torch.cuda.synchronize()
'''

t_kas_forward = benchmark.Timer(
    stmt=kas_test_forward,
    setup='from __main__ import kas_conv, torch',
    globals={'t_in': t_in})

torch_test_forward = '''
with torch.no_grad():
    torch_conv(t_in)
    torch.cuda.synchronize()
'''

t_torch_forward = benchmark.Timer(
    stmt=torch_test_forward,
    setup='from __main__ import torch_conv, torch',
    globals={'t_in': t_in})

print(t_kas_forward.timeit(100))
print(t_torch_forward.timeit(100))


t_in = torch.randn([64, 3, 128, 128], requires_grad=True, device=device)

kas_test_backward = '''
torch.sum(kas_conv(t_in)).backward()
torch.cuda.synchronize()
'''

t_kas_backward = benchmark.Timer(
    stmt=kas_test_backward,
    setup='from __main__ import kas_conv, torch',
    globals={'t_in': t_in})

torch_test_backward = '''
torch.sum(torch_conv(t_in)).backward()
torch.cuda.synchronize()
'''

t_torch_backward = benchmark.Timer(
    stmt=torch_test_backward,
    setup='from __main__ import torch_conv, torch',
    globals={'t_in': t_in})

print(t_kas_backward.timeit(10))
print(t_torch_backward.timeit(100))
