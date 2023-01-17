import os
import torch
import kas_cpp_bindings
from kas_cpp_bindings import Kernel

class KernelPack:
    def __init__(self, identifier: str, directory: str, name: str, size_params: list[int], inputs_shapes: list[list[int]], output_shape: list[int]):
        srcs = []
        filenames = [os.path.join(directory, name + '.pytorch.h'), os.path.join(directory, name + '_grad.pytorch.h')]
        for filename in filenames:
            with open(filename) as file:
                srcs.append(file.read())
        # JIT the compiled operator and load it.
        self._module = torch.utils.cpp_extension.load_inline(
            name=f'kas_{identifier}',
            cpp_sources=srcs,
            functions=[f'{name}_th_', f'{name}_grad_th_'],
            extra_cflags=['-std=c++17', '-g'],
            extra_ldflags=[f' -L{os.path.abspath(directory)} ',
                        ' -lcuda ',
                        f' -l:{name}.a '
                        f' -l:{name}_grad.a '],
            with_cuda=True,
            verbose=True
        )
        def forward(ctx, *args):
            out_forward = torch.empty(output_shape)
            self._module.forward(*size_params, *args, out_forward)
            ctx.save_for_backward(*args)
            return out_forward
        def backward(ctx, grad_output):
            grad_inputs = [torch.empty(shape) for shape in inputs_shapes]
            self._module.backward(*size_params, *ctx.saved_tensors, grad_output, *grad_inputs)
            return grad_inputs
        # Create an operator.
        self.KernelType = type(f'KASKernel{identifier}', (torch.autograd.Function,), {
            'forward': staticmethod(forward),
            'backward': staticmethod(backward),
        })

    def __del__(self):
        del self.KernelType
        del self._module
