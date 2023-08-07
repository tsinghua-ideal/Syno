import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from KAS import Assembled, Assembler, Sampler, Placeholder, CodeGenOptions, Path, Statistics


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.kernel_1 = Placeholder({"H": 32, "W": 32})
        self.kernel_2 = Placeholder({"H": 16, "W": 16})
        self.kernel_3 = Placeholder({"H": 16, "W": 16})

    def forward(self, x: torch.Tensor):
        x = self.kernel_1(x)
        x = x.view(1, 1, 32, 32)
        x = F.avg_pool2d(x, 2)
        x = x.view(16, 16)
        x = self.kernel_2(x)
        x = self.kernel_3(x)
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
    shared_s_2 = assembler.create_share(strided_window_H, w_s_2)
    # [H, s_2, W]

    # Create a ShiftOp.
    # shifted_W = assembler.create_shift(in_W, 1)
    # [H, s_2, W]

    # Create an ExpandOp.
    expanded_s_1 = assembler.create_expand(s_1)

    main_H.output(0) # Mark the H as output.
    in_W.output(1)
    # shifted_W.output(1) # Mark the W as output.
    shared_s_2.mean(0) # Mark the s_2 as sum reduction.
    expanded_s_1.mean(1) # Mark the s_1 as sum reduction.

    # Specify the input tensors.
    return assembler.assemble("simple_primitives_test", "in_0 * in_1", [in_H, in_W, expanded_s_1], [w_s_2])

def perform_trials(manual: bool):
    net = Model()
    sampler = Sampler(
        "[H,W]", "[H,W]", ["H: 2", "W: 2"], ["s_1=2: 2", "s_2=3: 2"], net=net, seed=42, depth=6,
        maximum_reductions=3,
        cuda=False, autoscheduler=CodeGenOptions.Adams2019,
        extra_options={
            "parallelism": "30",
            "shared_memory_limit_kb": "48",
            "shared_memory_sm_limit_kb": "64",
            "active_block_limit": "256",
            "active_warp_limit": "512",
        },
        num_worker_threads=16,
    )
    sampler._bind_debug_context()

    if not manual:
        Statistics.Print()
        print("Expanding...")
        sampler.root().expand(3)
        print("Completed expansion. Now searching...")
        while True:
            samples = sampler.random_final_nodes_with_prefix([], 256)
            print(f"Collected {len(samples)} samples.")
            if len(samples) > 0:
                node = samples[0]
                kernel = sampler.realize(net, node)
                if kernel.get_count_inputs() > 1:
                    kernel_packs = kernel.construct_kernel_packs()
                    print(f"Kernel files stored in {kernel.get_directory()}")
                    break
        Statistics.Print()
        print(node.get_nested_loops_as_final())
        print(node.get_composing_arcs())
        print(node.get_possible_path())
    else:
        assembler = sampler.create_assembler()
        node = manually_design(assembler)
        plausible_path = node.convert_to_path(sampler)
        print(f"Manually created {plausible_path}")
        sampler.visit(plausible_path) # Just to make sure it exists.
        kernel_packs = sampler.realize(net, node).construct_kernel_packs()
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
            param -= param.grad / max_grad * 1.0e-2
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
