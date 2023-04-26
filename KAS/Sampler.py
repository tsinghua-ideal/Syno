import os
import logging
import torch
from torch import nn
from typing import List
import kas_cpp_bindings
from kas_cpp_bindings import CodeGenOptions

from .KernelPack import KernelPack
from .Node import Next, Path, Node
from .Placeholder import Placeholder


class Sampler:
    """An immutable search tree. Use this to visit the tree built in C++."""

    @staticmethod
    def _extract_placeholders(net: nn.Module) -> List[Placeholder]:
        """Find all placeholders in the network. """
        placeholders: List[Placeholder] = [
            node for node in net.modules() if isinstance(node, Placeholder)]
        return placeholders

    def __init__(self, input_shape: str, output_shape: str, primary_specs: List[str], coefficient_specs: List[str], net: nn.Module = None, seed: int = 42, depth: int = 4, dim_lower: int = 2, dim_upper: int = 8, maximum_tensors = 2, save_path: str = './samples', cuda: bool = False, autoscheduler: CodeGenOptions.AutoScheduler = CodeGenOptions.AutoScheduler.ComputeRoot):
        options = kas_cpp_bindings.SampleOptions(
            seed=seed,
            depth=depth,
            dim_lower=dim_lower,
            dim_upper=dim_upper,
            maximum_tensors=maximum_tensors,
        )
        self._seed = seed
        self._save_path = save_path

        all_mappings = []
        if net is not None:
            placeholders = Sampler._extract_placeholders(net)
            if len(placeholders) == 0:
                raise ValueError('No placeholders found in the network.')
            all_mappings = [placeholder.mappings for placeholder in placeholders]

        self._sampler = kas_cpp_bindings.Sampler(
            input_shape, output_shape, primary_specs, coefficient_specs, all_mappings, options)
        self._codegen_options = kas_cpp_bindings.CodeGenOptions(
            cuda, autoscheduler)
        self._device = torch.device('cuda' if cuda else 'cpu')

    def root(self) -> Node:
        """Get the root node."""
        return self.visit([])

    def visit(self, path: Path) -> Node:
        """Visit a node via a path."""
        path = Path(path)
        return Node(path, self._sampler.visit(path.abs_path))

    def random_node_with_prefix(self, prefix: Path) -> Node:
        """Find a leaf node with specified prefix. Note that the Node is not necessarily final."""
        prefix = Path(prefix)
        path, node = self._sampler.random_node_with_prefix(prefix.abs_path)
        return Node(Path(path), node)

    def path_to_strs(self, path: Path) -> List[str]:
        node = self.root()
        strs = []
        for next in path:
            strs.append(next.description(node._node))
            node = node.get_child(next)
        return strs

    def _realize(self, node: Node) -> kas_cpp_bindings.Kernel:
        return node._realize_as_final(self._codegen_options)

    def realize(self, net: nn.Module, node: Node, identifier_prefix: str) -> List[KernelPack]:
        kernel = self._realize(node)
        logging.debug(f"Sampled kernel: {kernel}")
        save_path = os.path.join(self._save_path, identifier_prefix)
        os.makedirs(save_path, exist_ok=True)

        placeholders = Sampler._extract_placeholders(net)
        kernel_packs = []
        for i, placeholder in enumerate(placeholders):
            kernel_name = f'kernel_{i}'

            mappings = placeholder.mappings
            logging.debug(f"For kernel_{i} mappings: {mappings}")

            kernel.generate(save_path, kernel_name, mappings)

            inputs_shapes = kernel.get_inputs_shapes(mappings)
            logging.debug(f"Inputs shapes: {inputs_shapes}")
            output_shape = kernel.get_output_shape(mappings)
            logging.debug(f"Output shape: {output_shape}")

            identifier = identifier_prefix + "__" + str(i)
            kernel_packs.append(KernelPack(
                identifier, save_path, kernel_name, inputs_shapes, output_shape, self._device))

        return kernel_packs

    @staticmethod
    def replace(net: nn.Module, kernel_packs: List[KernelPack]):
        placeholders = Sampler._extract_placeholders(net)
        for placeholder, kernel_pack in zip(placeholders, kernel_packs):
            placeholder.reload(kernel_pack)
