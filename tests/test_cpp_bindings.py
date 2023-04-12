import logging
from kas_cpp_bindings import Sampler, SampleOptions, CodeGenOptions
import torch
import torch.utils.cpp_extension
import os

def test_sample():
    options = SampleOptions()
    sampler = Sampler("[H,W]", "[H,W]", ["H=16", "W=16"], ["s=2"], options)
    sampled_path = []
    trials = 0
    while not sampler.is_final(sampled_path):
        sampled_path = sampler.random_path_with_prefix([])
        trials += 1
    print(f"Found {sampled_path} after {trials} trials.")
    cg_opt = CodeGenOptions(False, CodeGenOptions.AutoScheduler.ComputeRoot)
    kernel = sampler.realize(sampled_path, cg_opt)
    kernel.generate("./save/py_kernel_simple", "kernel", {})

    # Load file
    srcs = []
    for filename in ['./save/py_kernel_simple/kernel.pytorch.h', './save/py_kernel_simple/kernel_grad.pytorch.h']:
        with open(filename) as file:
            srcs.append(file.read())

    # Compile
    module = torch.utils.cpp_extension.load_inline(
        name='test_inline_ext',
        cpp_sources=srcs,
        functions=['kernel_th_', 'kernel_grad_th_'],
        extra_cflags=['-std=c++17', '-g'],
        extra_ldflags=[f' -L{os.getcwd()}/save/py_kernel_simple ',
                       ' -lcuda ',
                       ' -l:kernel.a '
                       ' -l:kernel_grad.a '],
        with_cuda=True,
        verbose=True
    )

    inputs_shapes = kernel.get_inputs_shapes({})
    # The first item is the real input. The other are weights.
    inputs = [torch.randn(s) for s in inputs_shapes]
    output_tensor = torch.empty((16, 16), dtype=torch.float32)

    module.kernel_th_(*inputs, output_tensor)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_sample()
