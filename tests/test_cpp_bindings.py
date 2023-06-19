import logging
from KAS import KernelPack
from KAS.Bindings import Sampler, SampleOptions, CodeGenOptions, Loader
import torch
import torch.utils.cpp_extension
import os

def test_sample():
    options = SampleOptions()
    sampler = Sampler("[N,H,W]", "[N,H,W]", ["N=4: 0", "H=16", "W=16"], ["s=2"], [{}], [(0, 0)], options)
    trials = 0
    while True:
        _, sample = sampler.random_node_with_prefix([])
        trials += 1
        if sample.is_final():
            break
    print(f"Found {sample} after {trials} trials.")
    cg_opt = CodeGenOptions(False, CodeGenOptions.ComputeRoot)
    kernel = sample.realize_as_final([{}], cg_opt)
    kernel.generate_operator("./samples/py_kernel_simple", "kernel")
    kernel.generate_graphviz("./samples/py_kernel_simple", "kernel")

    # Load file
    loader = KernelPack.load_kernels("./samples/py_kernel_simple", "kernel", kernel.get_count_inputs(), 1, "cpu")

    inputs_shapes = kernel.get_inputs_shapes(True, 0)
    # The first item is the real input. The other are weights.
    inputs = [torch.randn(s) for s in inputs_shapes]
    output_tensor = torch.empty((4, 16, 16), dtype=torch.float32)

    loader.forward(0, *inputs, output_tensor)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_sample()
