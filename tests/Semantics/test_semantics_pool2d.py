import logging
import KAS
import torch
import torch.nn as nn
import os

def test_pool2d():
    device = torch.device("cuda:0")

    loader = KAS.KernelPack.load_kernels(
        directory=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../build/tests/Semantics/kernel_pool2d"),
        name="pool2d",
        count_inputs=1,
        count_kernels=1,
        device=device,
    )

    # Run CTests before this to generate the kernel
    pack = KAS.KernelPack(
        identifier="pool2d",
        loader=loader,
        index=0,
        unpadded_inputs_shapes=[[64, 3, 128, 128]],
        padded_inputs_shapes=[[64, 3, 130, 130]],
        unpadded_output_shape=[64, 3, 25, 25],
        padded_output_shape=[64, 3, 26, 26],
        device=device,
    )
    kas_pooling = KAS.Placeholder({})
    kas_pooling.reload(pack)
    torch_pooling = nn.AvgPool2d(5)

    t_in = torch.randn([64, 3, 128, 128], device=device)
    with torch.no_grad():
        t_out = kas_pooling(t_in)
        t_out_expected = torch_pooling(t_in)
        print("kas:", t_out.reshape(-1)[:10])
        print("torch:", t_out_expected.view(-1)[:10])
        # Actually we cannot make them close. The padding shifts the tiles by 1.
        print("forward is close:", torch.isclose(t_out[:,:,5:-5,5:-5], t_out_expected[:,:,5:-5,5:-5]).all())

    t_in = torch.randn([64, 3, 128, 128], requires_grad=True, device=device)
    torch.sum(kas_pooling(t_in)).backward()
    grad_kas = t_in.grad.detach()
    pack.zero_grad()
    t_in.grad = None
    torch.sum(torch_pooling(t_in)).backward()
    grad_torch = t_in.grad.detach()
    torch_pooling.zero_grad()
    t_in.grad = None
    print("grad_kas:", grad_kas.reshape(-1)[1500000:1500010])
    print("grad_torch:", grad_torch.view(-1)[1500000:1500010])
    backward_is_close = torch.isclose(grad_kas[:,:,5:-5,5:-5], grad_torch[:,:,5:-5,5:-5]).all()
    print("backward is close:", backward_is_close)
    assert backward_is_close


    import torch.utils.benchmark as benchmark

    t_in = torch.randn([64, 3, 128, 128], device=device)

    def kas_test_forward():
        with torch.no_grad():
            kas_pooling(t_in)
            torch.cuda.synchronize()

    t_kas_forward = benchmark.Timer(
        stmt="kas_test_forward()",
        globals={"kas_test_forward": kas_test_forward})

    def torch_test_forward():
        with torch.no_grad():
            torch_pooling(t_in)
            torch.cuda.synchronize()

    t_torch_forward = benchmark.Timer(
        stmt="torch_test_forward()",
        globals={"torch_test_forward": torch_test_forward})

    print(t_kas_forward.timeit(1000))
    print(t_torch_forward.timeit(1000))


    t_in = torch.randn([64, 3, 128, 128], requires_grad=True, device=device)

    def kas_test_backward():
        torch.sum(kas_pooling(t_in)).backward()
        torch.cuda.synchronize()

    t_kas_backward = benchmark.Timer(
        stmt="kas_test_backward()",
        globals={"kas_test_backward": kas_test_backward})

    def torch_test_backward():
        torch.sum(torch_pooling(t_in)).backward()
        torch.cuda.synchronize()

    t_torch_backward = benchmark.Timer(
        stmt="torch_test_backward()",
        globals={"torch_test_backward": torch_test_backward})

    print(t_kas_backward.timeit(100))
    print(t_torch_backward.timeit(100))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_pool2d()
