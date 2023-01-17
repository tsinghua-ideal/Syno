import os
import torch
from torch import nn
import kas_cpp_bindings
from kas_cpp_bindings import CodeGenOptions

from .KernelPack import KernelPack
from .Placeholder import Placeholder


class Sampler:
    def __init__(self, input_shape: str, output_shape: str, primary_specs: list[str], coefficient_specs: list[str], seed: int = 42, depth: int = 4, dim_lower: int = 2, dim_upper: int = 8, save_path: str = './save', cuda: bool = False, autoscheduler: CodeGenOptions.AutoScheduler = CodeGenOptions.AutoScheduler.Li2018):
        options = kas_cpp_bindings.SampleOptions(
            seed=seed,
            depth=depth,
            dim_lower=dim_lower,
            dim_upper=dim_upper,
        )
        self._save_path = save_path
        self._sampler = kas_cpp_bindings.Sampler(input_shape, output_shape, primary_specs, coefficient_specs, options)
        self._codegen_options = kas_cpp_bindings.CodeGenOptions(cuda, autoscheduler)

    # Here we simply replace the placeholders with sampled kernels. TODO: Add support for storing and loading kernels.
    def sample(self, net: nn.Module) -> list[int]:
        path = self._sampler.random_path_with_prefix([])
        kernel = self._sampler.realize(path)
        placeholders: list[Placeholder] = [node for node in net.modules() if isinstance(node, Placeholder)]
        identifier_prefix = '_'.join(map(str, path))
        save_path = os.path.join(self._save_path, identifier_prefix)
        os.makedirs(save_path, exist_ok=True)
        for i, placeholder in enumerate(placeholders):
            kernel_name = f'kernel_{i}'
            mappings = placeholder.mappings
            kernel.generate(save_path, kernel_name, self._codegen_options, mappings)
            identifier = identifier_prefix + "__" + str(i)
            kernel_args = kernel.get_arguments(mappings)
            inputs_shapes = kernel.get_inputs_shapes(mappings)
            output_shape = kernel.get_output_shape(mappings)
            placeholder.reload(KernelPack(identifier, save_path, kernel_name, kernel_args, inputs_shapes, output_shape))
        return path

