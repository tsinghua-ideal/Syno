import logging
from KAS import KernelPack
from KAS.Bindings import Sampler, SampleOptions, CodeGenOptions, Loader, LoaderArgs
import torch
import torch.utils.cpp_extension
import os

def test_sample():
    options = SampleOptions()
    sampler = Sampler("[N,H,W]", "[N,H,W]", ["N=4: 0", "H=16", "W=16"], ["s=2"], [{}], [(0, 0)], options)
    sampler.bind_debug_context()
    trials = 0
    while True:
        _, sample = sampler.random_node_with_prefix([])
        trials += 1
        if sample.is_final():
            break
    print(f"Found {sample} after {trials} trials.")
    cg_opt = CodeGenOptions(True, False, CodeGenOptions.ComputeRoot)
    kernel = sample.realize_as_final([{}], cg_opt, "./samples/py_kernel_simple", "kernel")

    # Load file
    args = kernel.get_loader_args()
    loader = Loader(LoaderArgs(
        args.path, args.symbol, args.cuda, args.count_inputs, args.count_kernels, args.valid_placeholder_indices
    ))

    inputs_shapes = kernel.get_inputs_shapes(True, 0)
    # The first item is the real input. The other are weights.
    inputs = [torch.randn(s) for s in inputs_shapes]
    output_tensor = torch.empty((4, 16, 16), dtype=torch.float32)

    loader.forward(0, *inputs, output_tensor)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_sample()
