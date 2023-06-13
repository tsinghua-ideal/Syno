import logging
import itertools
import os
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension
from torch import nn
from typing import List, Tuple, Union

from .Bindings import Loader


class KernelPack(nn.Module):
    """A wrapper for the generated kernel."""

    @staticmethod
    def load_kernels(directory: str, name: str, count_inputs: int, count_kernels: int, device: torch.device) -> Loader:
        """Load kernels from the given directory."""
        cuda = torch.device(device).type == 'cuda'
        return Loader(os.path.join(directory, name + ".so"), name, cuda, count_inputs, count_kernels)

    @staticmethod
    def get_paddings_from_shapes(unpadded_shape: List[int], padded_shape: List[int]) -> List[Union[None, Tuple[int, int]]]:
        """Get paddings from unpadded and padded shapes."""
        def get_paddings(unpadded: int, padded: int) -> slice:
            if padded == unpadded:
                return None
            elif padded > unpadded:
                left = (padded - unpadded) // 2
                right = padded - unpadded - left
                return (left, right)
            else:
                raise ValueError(
                    f'padded {padded} is smaller than unpadded {unpadded}.')
        return [get_paddings(unpadded, padded) for unpadded, padded in zip(unpadded_shape, padded_shape)]

    @staticmethod
    def pad_params(paddings: List[Union[None, Tuple[int, int]]]) -> Union[None, List[int]]:
        """Get parameters for nn.functional.pad."""
        if all(padding is None for padding in paddings):
            return None
        remaining_paddings = list(
            map(lambda x: x if x is not None else (0, 0), paddings))
        remaining_paddings.reverse()
        logging.info(f"paddings are {paddings}")
        logging.info(f"remaining_paddings are {remaining_paddings}")
        # flatten
        return list(itertools.chain.from_iterable(remaining_paddings))

    @staticmethod
    def crop_params(paddings: List[Union[None, Tuple[int, int]]]) -> Union[None, List[slice]]:
        """Get parameters for tensor slicing."""
        if all(padding is None for padding in paddings):
            return None
        return [slice(padding[0], -padding[1]) if padding is not None else slice(None) for padding in paddings]

    def __init__(self, identifier: str, loader: Loader, index: int, unpadded_inputs_shapes: List[List[int]], padded_inputs_shapes: List[List[int]], unpadded_output_shape: List[int], padded_output_shape: List[int], device: torch.device):
        super(KernelPack, self).__init__()

        self._loader = loader
        self._index = index

        # Collect inputs paddings.
        inputs_paddings = [self.get_paddings_from_shapes(
            unpadded_shape, padded_shape) for unpadded_shape, padded_shape in zip(unpadded_inputs_shapes, padded_inputs_shapes)]
        inputs_pad_params = [self.pad_params(
            paddings) for paddings in inputs_paddings]
        inputs_crop_params = [self.crop_params(
            paddings) for paddings in inputs_paddings]

        # Collect output paddings.
        output_paddings = self.get_paddings_from_shapes(
            unpadded_output_shape, padded_output_shape)
        output_pad_params = self.pad_params(output_paddings)
        output_crop_params = self.crop_params(output_paddings)

        # Forward kernel.
        def kernel_forward(ctx, *args):
            # The result that will be written to.
            out_forward = torch.empty(padded_output_shape, device=device)
            # Pad the inputs.
            args = tuple(
                x.contiguous() if pad_params is None else F.pad(x, pad_params).contiguous()
                for x, pad_params in zip(args, inputs_pad_params))
            # Call the operator.
            self._loader.forward(index, *args, out_forward)
            # Save for backward.
            ctx.save_for_backward(*args)
            # Crop if needed.
            return out_forward if output_crop_params is None else out_forward[output_crop_params]

        def kernel_backward(ctx, grad_output):
            # The result.
            grad_inputs = [torch.empty(shape, device=device)
                           for shape in padded_inputs_shapes]
            # Pad the output gradient.
            grad_output = grad_output.contiguous() if output_pad_params is None else F.pad(
                grad_output, output_pad_params).contiguous()
            # Call the operator.
            self._loader.backward(index,
                                  *ctx.saved_tensors, grad_output, *grad_inputs)
            # Crop if needed.
            return tuple(
                grad_input if crop_params is None else grad_input[crop_params]
                for grad_input, crop_params in zip(grad_inputs, inputs_crop_params))
        # Create an operator.
        self._Kernel = type(f'KASKernel{identifier}', (torch.autograd.Function,), {
            'forward': staticmethod(kernel_forward),
            'backward': staticmethod(kernel_backward),
        })

        # Initialize weights. Note that the first item is the input.
        # TODO: maybe we should add weight initializer?
        self.weights = nn.ParameterList([
            torch.nn.init.xavier_uniform_(
                torch.randn(shape, device=device)
            ) for shape in unpadded_inputs_shapes[1:]
        ])

    def forward(self, x):
        return self._Kernel.apply(x, *self.weights)

    def __del__(self):
        attr_dict = self.__dir__()
        if '_Kernel' in attr_dict:
            del self._Kernel
