import KAS
import torch
import torch.nn as nn
import os

device = torch.device("cuda:0")

# Run CTests before this to generate the kernel
pack = KAS.KernelPack("pooling", os.path.join(os.path.dirname(os.path.realpath(__file__)), "../build/tests/kernel_pooling"), "pooling", [[64, 3, 128, 128]], [64, 3, 25, 25], device)
kas_pooling = KAS.Placeholder({})
kas_pooling.reload(pack)
torch_pooling = nn.AvgPool2d(5)


t_in = torch.randn([64, 3, 128, 128], device=device)
with torch.no_grad():
    t_out = kas_pooling(t_in) / 25
    t_out_expected = torch_pooling(t_in)
    # torch.cuda.synchronize()
    print("kas:", t_out.view(-1)[:10])
    print("torch:", t_out_expected.view(-1)[:10])
    print("forward is close:", torch.isclose(t_out, t_out_expected).all())

t_in = torch.randn([64, 3, 128, 128], requires_grad=True, device=device)
torch.sum(kas_pooling(t_in)).backward()
grad_kas = t_in.grad.detach()
# torch.cuda.synchronize()
import time
time.sleep(1)
pack.zero_grad()
t_in.grad = None
torch.sum(torch_pooling(t_in)).backward()
grad_torch = t_in.grad.detach()
torch_pooling.zero_grad()
t_in.grad = None
# torch.cuda.synchronize()
print("grad_kas:", grad_kas.view(-1)[1500000:1500010])
print("grad_torch:", grad_torch.view(-1)[1500000:1500010])
print("backward is close:", torch.isclose(grad_kas / 25, grad_torch).all())


import torch.utils.benchmark as benchmark

t_in = torch.randn([64, 3, 128, 128], device=device)

kas_test_forward = '''
with torch.no_grad():
    kas_pooling(t_in)
    torch.cuda.synchronize()
'''

t_kas_forward = benchmark.Timer(
    stmt=kas_test_forward,
    setup='from __main__ import kas_pooling, torch',
    globals={'t_in': t_in})

torch_test_forward = '''
with torch.no_grad():
    torch_pooling(t_in)
    torch.cuda.synchronize()
'''

t_torch_forward = benchmark.Timer(
    stmt=torch_test_forward,
    setup='from __main__ import torch_pooling, torch',
    globals={'t_in': t_in})

print(t_kas_forward.timeit(1000))
print(t_torch_forward.timeit(1000))


t_in = torch.randn([64, 3, 128, 128], requires_grad=True, device=device)

kas_test_backward = '''
torch.sum(kas_pooling(t_in)).backward()
torch.cuda.synchronize()
'''

t_kas_backward = benchmark.Timer(
    stmt=kas_test_backward,
    setup='from __main__ import kas_pooling, torch',
    globals={'t_in': t_in})

torch_test_backward = '''
torch.sum(torch_pooling(t_in)).backward()
torch.cuda.synchronize()
'''

t_torch_backward = benchmark.Timer(
    stmt=torch_test_backward,
    setup='from __main__ import torch_pooling, torch',
    globals={'t_in': t_in})

print(t_kas_backward.timeit(100))
print(t_torch_backward.timeit(100))
