import logging
import itertools
import importlib
import os
import tarfile
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension
from torch import nn
from typing import List, Tuple, Union

from . import Bindings


class KernelPack(nn.Module):
    """A wrapper for the generated kernel."""

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

    def __init__(self, identifier: str, loader: Bindings.Loader, index: int, unpadded_inputs_shapes: List[List[int]], padded_inputs_shapes: List[List[int]], unpadded_output_shape: List[int], padded_output_shape: List[int], device: torch.device):
        super(KernelPack, self).__init__()

        self._loader = loader
        self._index = index
        self._unpadded_inputs_shapes = unpadded_inputs_shapes
        self._padded_inputs_shapes = padded_inputs_shapes
        self._unpadded_output_shape = unpadded_output_shape
        self._padded_output_shape = padded_output_shape

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

        # Create weights (except the first input)
        self.weights = nn.ParameterList([torch.zeros(shape, device=device) for shape in unpadded_inputs_shapes[1:]])

    def forward(self, x):
        return self._Kernel.apply(x, *self.weights)

    def __del__(self):
        attr_dict = self.__dir__()
        if '_Kernel' in attr_dict:
            del self._Kernel

KAS_KERNEL_LOADER_COUNTER = 0

class KernelLoader:
    """All the metadata of a generated KAS kernel."""

    def __init__(self, kernel: Bindings.Kernel) -> None:
        global KAS_KERNEL_LOADER_COUNTER
        self._kernel = kernel
        self._identifier = f'KASKernel_{KAS_KERNEL_LOADER_COUNTER}'
        KAS_KERNEL_LOADER_COUNTER += 1

    @staticmethod
    def from_directory(directory: os.PathLike) -> 'KernelLoader':
        return KernelLoader(Bindings.Kernel(directory))

    def archive_to(self, file: os.PathLike, overwrite: bool = True) -> None:
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if not os.path.exists(file) or overwrite:
            with tarfile.open(file, "w:gz") as tar:
                tar.add(self.get_directory(), "kernel_dir")

    def get_name(self) -> str:
        return self._kernel.get_name()

    def get_directory(self) -> os.PathLike:
        return self._kernel.get_directory()

    def halide(self) -> bool:
        return self._kernel.halide()

    def get_device(self) -> bool:
        if self._kernel.use_cuda():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def get_count_placeholders(self) -> int:
        return self._kernel.get_count_placeholders()

    def get_count_valid_kernels(self) -> int:
        return self._kernel.get_count_valid_kernels()

    def get_valid_placeholder_index(self, index: int) -> int:
        return self._kernel.get_valid_placeholder_index(index)

    def get_count_inputs(self) -> int:
        return self._kernel.get_count_inputs()

    def get_inputs_shapes(self, padded: bool, index: int) -> List[List[int]]:
        return self._kernel.get_inputs_shapes(padded, index)

    def get_output_shape(self, padded: bool, index: int) -> List[int]:
        return self._kernel.get_output_shape(padded, index)

    def get_consts(self, index: int) -> str:
        return self._kernel.get_consts(index)

    def get_flops(self, index: int) -> int:
        return self._kernel.get_flops(index)

    def get_total_flops(self) -> int:
        return self._kernel.get_total_flops()

    def _get_loader_args(self) -> Bindings.LoaderArgs:
        args = self._kernel.get_loader_args()
        return Bindings.LoaderArgs(
            args.path, args.symbol, args.cuda, args.count_inputs, args.count_kernels, args.valid_placeholder_indices
        )

    # Construct kernel packs with globally unique identifier. The identifier can be arbitrary, because we are constructing Python classes dynamically.
    def construct_kernel_packs(self) -> List[KernelPack]:
        logging.debug(f"Constructing kernel:\n{self._kernel}")
        logging.debug(f"Total FLOPs: {self.get_total_flops()}")

        if not self.halide():
            # No Halide! Just load the PyTorch modules.
            pytorch_modules_file = os.path.join(self.get_directory(), "kernels.py")
            logging.debug(f"Loading PyTorch modules from {pytorch_modules_file}")
            spec = importlib.util.spec_from_file_location("kernels", pytorch_modules_file)
            kernels = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(kernels)
            device = self.get_device()
            kernel_packs = []
            kernel_name_prefix = self.get_name()
            for i in range(self.get_count_placeholders()):
                logging.debug(f"For placeholder {i},")
                logging.debug(f"Consts: {self.get_consts(i)}")
                logging.debug(f"FLOPs: {self.get_flops(i)}")
                valid_i = self.get_valid_placeholder_index(i)
                kernel_name = f"{kernel_name_prefix}_{valid_i}"
                kernel_packs.append(getattr(kernels, kernel_name)(i).to(device))
            return kernel_packs

        loader = Bindings.Loader(self._get_loader_args())

        kernel_packs = []
        for i in range(self.get_count_placeholders()):
            logging.debug(f"For placeholder {i},")

            logging.debug(f"Consts: {self.get_consts(i)}")
            logging.debug(f"FLOPs: {self.get_flops(i)}")

            unpadded_inputs_shapes = self.get_inputs_shapes(False, i)
            padded_inputs_shapes = self.get_inputs_shapes(True, i)
            logging.debug(
                f"Unpadded inputs shapes: {unpadded_inputs_shapes}, padded inputs shapes: {padded_inputs_shapes}")
            unpadded_output_shape = self.get_output_shape(False, i)
            padded_output_shape = self.get_output_shape(True, i)
            logging.debug(
                f"Unpadded output shape: {unpadded_output_shape}, padded output shape: {padded_output_shape}")

            identifier = f"{self._identifier}__{i}"
            kernel_packs.append(KernelPack(
                identifier=identifier,
                loader=loader,
                index=i,
                unpadded_inputs_shapes=unpadded_inputs_shapes,
                padded_inputs_shapes=padded_inputs_shapes,
                unpadded_output_shape=unpadded_output_shape,
                padded_output_shape=padded_output_shape,
                device=self.get_device()))

        return kernel_packs

    def __repr__(self) -> str:
        return f"{self._kernel}"
