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

def test_conv2d():
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
    torch_conv = nn.Conv2d(size_C_in, size_C_out, (size_K, size_K), bias=False, padding="same", padding_mode='zeros', device=device)

    pack.weights = nn.ParameterList([torch.randn([size_C_out, size_C_in, size_K, size_K], device=device)])
    torch_conv.weight = nn.Parameter(pack.weights[0].detach())


    assert torch.isclose(pack.weights[0], torch_conv.weight).all()
    t_in = torch.randn([size_N, size_C_in, size_H, size_W], device=device)

    with torch.no_grad():
        k_out = kas_conv(t_in)
        t_out = torch_conv(t_in)
        print("forward_kas:", k_out.view(-1)[size_output // 2 : size_output // 2 + 10])
        print("forward_torch:", t_out.view(-1)[size_output // 2 : size_output // 2 + 10])
        forward_is_close = torch.isclose(k_out, t_out, atol=1e-2).all()
        print("forward is close:", forward_is_close)
        # assert forward_is_close

    t_in = torch.randn([size_N, size_C_in, size_H, size_W], requires_grad=True, device=device)
    torch.sum(kas_conv(t_in)).backward()
    grad_kas = t_in.grad.detach()
    pack.zero_grad(True)
    t_in.grad = None
    torch.sum(torch_conv(t_in)).backward()
    grad_torch = t_in.grad.detach()
    torch_conv.zero_grad(True)
    t_in.grad = None
    print("grad_kas:", grad_kas.view(-1)[size_input // 2 : size_input // 2 + 10])
    print("grad_torch:", grad_torch.view(-1)[size_input // 2 : size_input // 2 + 10])
    backward_is_close = torch.isclose(grad_kas, grad_torch, atol=1e-2).all()
    print("backward is close:", backward_is_close)
    assert backward_is_close


    import torch.utils.benchmark as benchmark

    t_in = torch.randn([size_N, size_C_in, size_H, size_W], device=device)

    def kas_test_forward():
        with torch.no_grad():
            kas_conv(t_in)
            torch.cuda.synchronize()

    t_kas_forward = benchmark.Timer(
        stmt="kas_test_forward()",
        globals={"kas_test_forward": kas_test_forward})

    def torch_test_forward():
        with torch.no_grad():
            torch_conv(t_in)
            torch.cuda.synchronize()

    t_torch_forward = benchmark.Timer(
        stmt="torch_test_forward()",
        globals={"torch_test_forward": torch_test_forward})

    print(t_kas_forward.timeit(100))
    print(t_torch_forward.timeit(100))


    t_in = torch.randn([size_N, size_C_in, size_H, size_W], requires_grad=True, device=device)

    def kas_test_backward():
        torch.sum(kas_conv(t_in)).backward()
        torch.cuda.synchronize()

    t_kas_backward = benchmark.Timer(
        stmt="kas_test_backward()",
        globals={"kas_test_backward": kas_test_backward})

    def torch_test_backward():
        torch.sum(torch_conv(t_in)).backward()
        torch.cuda.synchronize()

    t_torch_backward = benchmark.Timer(
        stmt="torch_test_backward()",
        globals={"torch_test_backward": torch_test_backward})

    print(t_kas_backward.timeit(10))
    print(t_torch_backward.timeit(100))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_conv2d()
