import os
import torch
import torch.utils.cpp_extension
from torch import nn
import kas_cpp_bindings
from kas_cpp_bindings import Kernel

class KernelPack(nn.Module):
    def __init__(self, identifier: str, directory: str, name: str, size_params: list[int], inputs_shapes: list[list[int]], output_shape: list[int]):
        super(KernelPack, self).__init__()

        srcs = []
        filenames = [os.path.join(directory, name + '.pytorch.h'), os.path.join(directory, name + '_grad.pytorch.h')]
        for filename in filenames:
            with open(filename) as file:
                srcs.append(file.read())

        # JIT the compiled operator and load it.
        forward_name = f'{name}_th_'
        backward_name = f'{name}_grad_th_'
        self._module = torch.utils.cpp_extension.load_inline(
            name=f'kas_{identifier}',
            cpp_sources=srcs,
            functions=[forward_name, backward_name],
            extra_cflags=['-std=c++17', '-g'],
            extra_ldflags=[f' -L{os.path.abspath(directory)} ',
                        ' -lcuda ',
                        f' -l:{name}.a '
                        f' -l:{name}_grad.a '],
            with_cuda=True,
            verbose=True
        )

        def kernel_forward(ctx, *args):
            out_forward = torch.empty(output_shape)
            args = tuple(map(lambda x: x.contiguous(), args))
            getattr(self._module, forward_name)(*size_params, *args, out_forward)
            ctx.save_for_backward(*args)
            return out_forward
        def kernel_backward(ctx, grad_output):
            grad_inputs = [torch.empty(shape) for shape in inputs_shapes]
            grad_output = grad_output.contiguous()
            getattr(self._module, backward_name)(*size_params, *ctx.saved_tensors, grad_output, *grad_inputs)
            return tuple(grad_inputs)
        # Create an operator.
        self._Kernel = type(f'KASKernel{identifier}', (torch.autograd.Function,), {
            'forward': staticmethod(kernel_forward),
            'backward': staticmethod(kernel_backward),
        })

        # Initialize weights. Note that the first item is the input.
        self.weights = nn.ParameterList([torch.randn(shape) for shape in inputs_shapes[1:]])

    def forward(self, x):
        return self._Kernel.apply(x, *self.weights)

    def __del__(self):
        del self._Kernel
        del self._module
