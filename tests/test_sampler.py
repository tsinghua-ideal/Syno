import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from KAS import Assembled, Assembler, Sampler, Placeholder, CodeGenOptions, Path


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.kernel_1 = Placeholder({"H": 32, "W": 32})
        self.kernel_2 = Placeholder({"H": 16, "W": 16})

    def forward(self, x: torch.Tensor):
        x = self.kernel_1(x)
        x = x.view(1, 1, 32, 32)
        x = F.avg_pool2d(x, 2)
        x = x.view(16, 16)
        x = self.kernel_2(x)
        return x


def manually_design(assembler: Assembler) -> Assembled:
    # First obtain the sizes. Note that the sizes can do arithmetics, i.e., you can H * W or H / s_1.
    H, W, s_1, s_2 = assembler.get_sizes("H", "W", "s_1", "s_2")

    # Next create input dimensions. We will perform transforms on them.
    # The inputs are: [H, W], [s_2].
    in_H, in_W, w_s_2 = assembler.make_dims_of_sizes(H, W, s_2)
    # They are blended into [H, W, s_2].

    # Create an UnfoldOp.
    main_H, window_H = assembler.create_unfold(in_H, s_1 * s_2)
    # Now, [H, s_1 * s_2, W, s_2]

    # Create a StrideOp.
    strided_window_H = assembler.create_stride(window_H, s_1)
    # [H, s_2, W, s_2]

    # Create a ShareOp.
    shared_s_2 = assembler.create_share(w_s_2, strided_window_H)
    # [H, s_2, W]

    # Create a ShiftOp.
    shifted_W = assembler.create_shift(in_W, 1)
    # [H, s_2, W]

    main_H.output(0) # Mark the H as output.
    shifted_W.output(1) # Mark the W as output.
    shared_s_2.sum(0) # Mark the s_2 as sum reduction.

    # Specify the input tensors.
    return assembler.assemble([in_H, in_W], [w_s_2])

def perform_trials(manual: bool):
    net = Model()
    sampler = Sampler("[H,W]", "[H,W]", [], ["s_1=2", "s_2=3"], net=net, seed=42, depth=8,
                      cuda=False, autoscheduler=CodeGenOptions.ComputeRoot)

    if not manual:
        while True:
            node = sampler.random_node_with_prefix(Path([]))
            if node.is_final():
                kernel_packs, total_flops = sampler.realize(net, node, "test_sampler")
                if total_flops > 0:
                    break
    else:
        assembler = sampler.create_assembler()
        node = manually_design(assembler)
        print(f"Manually created {node.convert_to_path(sampler)}")
        kernel_packs, _ = sampler.realize(net, node, "test_sampler")
    sampler.replace(net, kernel_packs)

    in_tensor = torch.randn((32, 32), requires_grad=True)
    out_tensor = net(in_tensor)
    print("First output:", out_tensor)
    target = torch.randn((16, 16))
    # compute gradient
    loss = F.mse_loss(out_tensor, target)
    print("First loss:", loss)
    loss.backward()
    # descent
    with torch.no_grad():
        # get max component
        max_grad = max(torch.max(param.grad) for param in net.parameters())
        for param in net.parameters():
            param -= param.grad / max_grad * 1.0e-3
            param.grad.zero_()
    out_tensor = net(in_tensor)
    print("Second output:", out_tensor)
    new_loss = F.mse_loss(out_tensor, target)
    print("Second loss:", new_loss)
    assert new_loss <= loss


def test_sampler():
    perform_trials(manual=False)
    perform_trials(manual=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_sampler()
