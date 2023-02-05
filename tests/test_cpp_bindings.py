import logging
from kas_cpp_bindings import *
import torch
import torch.utils.cpp_extension
import os

def test_sample():
    options = SampleOptions()
    sampler = Sampler("[H,W]", "[N,C,H,W]", [], [], options)
    print(sampler.random_path_with_prefix([]))
    assert sampler.is_final([0])
    kernel = sampler.realize([0])
    cg_opt = CodeGenOptions(False, CodeGenOptions.AutoScheduler.ComputeRoot)
    kernel.generate("build/py_kernel_simple", "kernel", cg_opt, {"H": 2, "W": 2, "N": 2, "C": 2})

    # Load file
    srcs = []
    for filename in ['build/py_kernel_simple/kernel.pytorch.h', 'build/py_kernel_simple/kernel_grad.pytorch.h']:
        with open(filename) as file:
            srcs.append(file.read())

    # Compile
    module = torch.utils.cpp_extension.load_inline(
        name='test_inline_ext',
        cpp_sources=srcs,
        functions=['kernel_th_', 'kernel_grad_th_'],
        extra_cflags=['-std=c++17', '-g'],
        extra_ldflags=[f' -L{os.getcwd()}/build/py_kernel_simple ',
                       ' -lcuda ',
                       ' -l:kernel.a '
                       ' -l:kernel_grad.a '],
        with_cuda=True,
        verbose=True
    )

    kernel_args = kernel.get_arguments({})
    inputs_shapes = kernel.get_inputs_shapes({})
    # The first item is the real input. The other are weights.
    inputs = [torch.randn(s) for s in inputs_shapes]
    output_tensor = torch.randn((2, 2, 2, 2))

    module.kernel_th_(*kernel_args, *inputs, output_tensor)
    
    print(' * '.join([str(input_tensor) for input_tensor in inputs]), '=', output_tensor)
    computed = torch.einsum('ij, kl -> ijkl', *inputs)
    assert torch.isclose(output_tensor, computed).all()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_sample()
