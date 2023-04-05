import os
import logging
import torch
from torch import nn
import kas_cpp_bindings
from kas_cpp_bindings import CodeGenOptions

from .KernelPack import KernelPack
from .Placeholder import Placeholder


class Sampler:
    def __init__(self, input_shape: str, output_shape: str, primary_specs: list[str], coefficient_specs: list[str], seed: int = 42, depth: int = 4, dim_lower: int = 2, dim_upper: int = 8, save_path: str = './save', cuda: bool = False, autoscheduler: CodeGenOptions.AutoScheduler = CodeGenOptions.AutoScheduler.ComputeRoot):
        options = kas_cpp_bindings.SampleOptions(
            seed=seed,
            depth=depth,
            dim_lower=dim_lower,
            dim_upper=dim_upper,
        )
        self._seed = seed
        self._save_path = save_path
        self._sampler = kas_cpp_bindings.Sampler(input_shape, output_shape, primary_specs, coefficient_specs, options)
        self._codegen_options = kas_cpp_bindings.CodeGenOptions(cuda, autoscheduler)
        self._device = torch.device('cuda' if cuda else 'cpu')

    def _path_str(self, path: list[int]) -> str:
        prefixes = [path[:l] for l in range(1, len(path) + 1)]
        layers = [self._sampler.node_str([])] + [s for prefix in prefixes for s in [
            self._sampler.op_str(prefix),
            self._sampler.node_str(prefix),
        ]]
        layers.reverse()
        return '\n'.join(layers)

    def _realize(self, path: list[int]) -> kas_cpp_bindings.Kernel:
        return self._sampler.realize(path, self._codegen_options)

    # Here we simply replace the placeholders with sampled kernels. TODO: Add support for storing and loading kernels.
    def sample(self, net: nn.Module, prefix: list[int] = []) -> list[int]:
        path = self._sampler.random_path_with_prefix(prefix)
        # TODO: what if the sample fails?
        kernel = self._realize(path)
        logging.debug(f"Sampled kernel: {kernel}")
        logging.debug(f"Path: {self._path_str(path)}")
        placeholders: list[Placeholder] = [node for node in net.modules() if isinstance(node, Placeholder)]
        identifier_prefix = '_'.join(map(str, path))
        save_path = os.path.join(self._save_path, identifier_prefix)
        os.makedirs(save_path, exist_ok=True)
        for i, placeholder in enumerate(placeholders):
            kernel_name = f'kernel_{i}'
            mappings = placeholder.mappings
            logging.debug(f"For kernel_{i} mappings: {mappings}")
            kernel.generate(save_path, kernel_name, mappings)
            identifier = identifier_prefix + "__" + str(i)
            inputs_shapes = kernel.get_inputs_shapes(mappings)
            logging.debug(f"Inputs shapes: {inputs_shapes}")
            output_shape = kernel.get_output_shape(mappings)
            logging.debug(f"Output shape: {output_shape}")
            placeholder.reload(KernelPack(identifier, save_path, kernel_name, inputs_shapes, output_shape, self._device))
        return path

    def is_dead_end(self, path: list[int]) -> bool:
        return (not self._sampler.is_final(path)) and self._sampler.children_count(path) == 0

    def is_terminal(self, path: list[int]) -> bool:
        # Either the path is a final node, or it is a dead end.
        return self._sampler.is_final(path) or self._sampler.children_count(path) == 0

    def children_count(self, path: list[int]) -> int:
        if self.is_terminal(path):
            raise ValueError("Cannot get children count of a terminal node.")
        return self._sampler.children_count(path)

    def children_types(self, path: list[int]) -> list[str]:
        if self.is_terminal(path):
            raise ValueError("Cannot get children types of a terminal node.")
        return self._sampler.children_types(path)
