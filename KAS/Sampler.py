import os
import logging
import torch
from torch import nn
from typing import Any, List, Dict, Optional, Tuple, Union
from KAS import Bindings
from KAS.Assembler import Assembled

from KAS.Node import Node, PseudoPath, VisitedNode

from . import Bindings
from .Assembler import Assembled, Assembler
from .Bindings import CodeGenOptions
from .KernelPack import KernelPack
from .Node import (
    Next,
    PseudoNext,
    Path,
    PseudoPath,
    Node,
    VisitedNode,
    MockNodeMetadata,
    MockNode,
    MockVisitedNode,
)
from .Placeholder import Placeholder


class Sampler:
    """An immutable search tree. Use this to visit the tree built in C++."""

    @staticmethod
    def _extract_placeholders(net: nn.Module) -> List[Placeholder]:
        """Find all placeholders in the network."""
        placeholders: List[Placeholder] = [
            node for node in net.modules() if isinstance(node, Placeholder)]
        return placeholders

    @staticmethod
    def _get_all_mappings(placeholders: List[Placeholder]) -> List[Dict[str, int]]:
        if len(placeholders) == 0:
            logging.warning('No placeholders found in the network.')
        return [placeholder.mappings for placeholder in placeholders]

    @staticmethod
    def _extract_all_mappings(net: nn.Module) -> List[Dict[str, int]]:
        placeholders = Sampler._extract_placeholders(net)
        return Sampler._get_all_mappings(placeholders)

    @staticmethod
    def replace(net: nn.Module, kernel_packs: List[KernelPack]):
        placeholders = Sampler._extract_placeholders(net)
        for placeholder, kernel_pack in zip(placeholders, kernel_packs):
            placeholder.reload(kernel_pack)

    def __init__(self, input_shape: str, output_shape: str, primary_specs: List[str], coefficient_specs: List[str], net: nn.Module = None, fixed_io_pairs: List[Tuple[int, int]] = [], seed: int = 42, depth: int = 4, dim_lower: int = 2, dim_upper: int = 8, maximum_tensors: int = 2, maximum_reductions: int = 2, max_flops = 1e15, maximum_variables_in_size: int = 16, maximum_variables_powers_in_size: int = 16, max_strided_dim_size: int = 30, max_unfold_kernel_size: int = 30, minimum_unfold_ratio: float = 2.0, minimum_merge_ratio: float = 2.0, disallow_discontinuous_view: bool = True, canonicalize_unfold_order: bool = True, maximum_merges: int = -1, maximum_splits: int = -1, maximum_shifts: int = -1, maximum_strides: int = -1, maximum_unfolds: int = -1, maximum_shares: int = -1, save_path: str = './samples', cuda: bool = False, autoscheduler: CodeGenOptions.AutoScheduler = CodeGenOptions.AutoScheduler.ComputeRoot, rfactor_threshold: int = 32, in_bounds_likely_threshold: float = 0.3):
        options = Bindings.SampleOptions(
            seed=seed,
            depth=depth,
            dim_lower=dim_lower,
            dim_upper=dim_upper,
            maximum_tensors=maximum_tensors,
            maximum_reductions=maximum_reductions,
            max_flops=max_flops,
            maximum_variables_in_size=maximum_variables_in_size,
            maximum_variables_powers_in_size=maximum_variables_powers_in_size,
            max_strided_dim_size=max_strided_dim_size,
            max_unfold_kernel_size=max_unfold_kernel_size,
            minimum_unfold_ratio=minimum_unfold_ratio,
            minimum_merge_ratio=minimum_merge_ratio,
            disallow_discontinuous_view=disallow_discontinuous_view,
            canonicalize_unfold_order=canonicalize_unfold_order,
            maximum_merges=maximum_merges,
            maximum_splits=maximum_splits,
            maximum_shifts=maximum_shifts,
            maximum_strides=maximum_strides,
            maximum_unfolds=maximum_unfolds,
            maximum_shares=maximum_shares,
        )
        self._seed = seed
        self._save_path = save_path

        all_mappings = []
        if net is not None:
            all_mappings = Sampler._extract_all_mappings(net)

        self._sampler = Bindings.Sampler(
            input_shape, output_shape, primary_specs, coefficient_specs, all_mappings, fixed_io_pairs, options)
        self._codegen_options = Bindings.CodeGenOptions(
            use_gpu=cuda,
            auto_scheduler=autoscheduler,
            rfactor_threshold=rfactor_threshold,
            in_bounds_likely_threshold=in_bounds_likely_threshold,
        )
        self._device = torch.device('cuda' if cuda else 'cpu')

    def root(self) -> VisitedNode:
        """Get the root node."""
        return self.visit([])

    def visit(self, path: PseudoPath) -> Optional[VisitedNode]:
        """Visit a node via a path."""
        path = Path(path)
        visited_node = self._sampler.visit(path.abs_path)
        if visited_node is None:
            return None
        return VisitedNode(path, visited_node)

    def random_node_with_prefix(self, prefix: PseudoPath) -> Optional[VisitedNode]:
        """Find a leaf node with specified prefix. Note that the Node is not necessarily final."""
        prefix = Path(prefix)
        visited_node = self._sampler.random_node_with_prefix(prefix.abs_path)
        if visited_node is None:
            return None
        path, node = visited_node
        return VisitedNode(Path(path), node)

    def path_to_strs(self, path: PseudoPath) -> List[str]:
        node = self.root()
        strs = []
        for next in path:
            strs.append(node.get_child_description(next))
            node = node.get_child(next)
        return strs

    def _realize(self, node: Union[Node, Assembled], all_mappings: List[Dict[str, int]]) -> Bindings.Kernel:
        if isinstance(node, Assembled):
            return node._realize(all_mappings, self._codegen_options)
        return node._realize_as_final(all_mappings, self._codegen_options)

    def realize(self, net: nn.Module, node: Union[Node, Assembled], identifier_prefix: str) -> Tuple[List[KernelPack], int]:
        placeholders = Sampler._extract_placeholders(net)
        all_mappings = Sampler._get_all_mappings(placeholders)

        kernel = self._realize(node, all_mappings)
        total_flops = kernel.get_total_flops()
        logging.debug(f"Realizing kernel:\n{kernel}")
        logging.debug(f"Total FLOPs: {total_flops}")
        save_path = os.path.join(self._save_path, identifier_prefix)
        # if os.path.exists(save_path):
        #     shutil.rmtree(save_path)
        os.makedirs(save_path, exist_ok=True)

        if isinstance(node, Assembled):
            kernel_name_prefix = f'kernel_manual'
        else:
            kernel_name_prefix = f'kernel_{abs(hash(node.to_node()))}'
        logging.debug("Generating kernel files...")
        kernel.generate_graphviz(save_path, kernel_name_prefix)
        kernel.generate_operator(save_path, kernel_name_prefix)
        logging.debug("Successfully generated kernel files.")
        loader = KernelPack.load_kernels(
            save_path, kernel_name_prefix, kernel.get_count_inputs(), len(placeholders), self._device)

        kernel_packs = []
        for i in range(len(placeholders)):
            kernel_name = f'{kernel_name_prefix}_{i}'
            logging.debug(f"For placeholder {i},")

            logging.debug(f"Consts: {kernel.get_consts(i)}")
            logging.debug(f"FLOPs: {kernel.get_flops(i)}")
            placeholders[i].set_flops(kernel.get_flops(i))

            unpadded_inputs_shapes = kernel.get_inputs_shapes(False, i)
            padded_inputs_shapes = kernel.get_inputs_shapes(True, i)
            logging.debug(
                f"Unpadded inputs shapes: {unpadded_inputs_shapes}, padded inputs shapes: {padded_inputs_shapes}")
            unpadded_output_shape = kernel.get_output_shape(False, i)
            padded_output_shape = kernel.get_output_shape(True, i)
            logging.debug(
                f"Unpadded output shape: {unpadded_output_shape}, padded output shape: {padded_output_shape}")

            identifier = identifier_prefix + "__" + str(i)
            kernel_packs.append(KernelPack(
                identifier=identifier,
                loader=loader,
                index=i,
                unpadded_inputs_shapes=unpadded_inputs_shapes,
                padded_inputs_shapes=padded_inputs_shapes,
                unpadded_output_shape=unpadded_output_shape,
                padded_output_shape=padded_output_shape,
                device=self._device))

        return kernel_packs, total_flops

    def create_assembler(self):
        return Assembler(self._sampler.create_assembler())

    def _bind_debug_context(self):
        self._sampler.bind_debug_context()


VertexIndex = Union[int, str]  # Can be index in the vertices list or name.


class MockSampler(Sampler):
    """Stores a DAG of nodes."""

    def __init__(
        self,
        vertices: List[Union[Dict[str, Any], str]],
        edges: List[Tuple[VertexIndex, List[Tuple[PseudoNext, VertexIndex]]]],
        seed: int=0xdeadbeaf
    ) -> None:
        """Initialize a MockSampler.
        The first vertex is the root.
        Example:
        ```
            vertices = ['root', 'a', 'b', {'name': 'final', 'is_final': True, 'accuracy': 0.9, 'arbitrary_metadata': 42}]
            edges = [
                ('root', [('Share(0)', 'a'), ('Share(1)', 'b')]),
                ('a', [('Share(2)', 'final')]),
                ('b', [('Share(3)', 'final')]),
            ]
            sampler = MockSampler(vertices, edges)
        ```
        """
        self._mock_nodes = [
            MockNodeMetadata(self, id, **(v if isinstance(v, dict) else {"name": str(v)}))
            for id, v in enumerate(vertices)
        ]
        self._mock_name_to_id = {
            str(mock_node): mock_node.mock_get_id() for mock_node in self._mock_nodes
        }

        def index_of(vertex_index: VertexIndex) -> int:
            if isinstance(vertex_index, int):
                return vertex_index
            return self._mock_name_to_id[vertex_index]

        def process_edges(
            pairs: List[Tuple[PseudoNext, VertexIndex]]
        ) -> Dict[Next, MockNodeMetadata]:
            return {
                Path.to_next(next): self._mock_nodes[index_of(vertex_index)]
                for next, vertex_index in pairs
            }

        mock_edges = {index_of(src): process_edges(dst) for src, dst in edges}
        self._mock_edges = {
            index: mock_edges.get(index, {}) for index in range(len(self._mock_nodes))
        }
        
        self._seed = seed

    def mock_get_children(self, id: int) -> Dict[Next, MockNodeMetadata]:
        return self._mock_edges[id]

    def visit(self, path: PseudoPath) -> Optional[MockVisitedNode]:
        path = Path(path)
        node = self._mock_nodes[0]
        for next in path:
            children = self._mock_edges[node.mock_get_id()]
            if next not in children:
                return None
            node = self._mock_edges[node.mock_get_id()][next]
        return MockVisitedNode(path, node)

    def random_node_with_prefix(self, prefix: PseudoPath) -> Optional[VisitedNode]:
        raise NotImplementedError()

    def _realize(self, node: Union[Node, Assembled], all_mappings: List[Dict[str, int]]) -> Bindings.Kernel:
        raise NotImplementedError()

    def realize(self, net: nn.Module, node: Union[Node, Assembled], identifier_prefix: str) -> Tuple[List[KernelPack], int]:
        raise NotImplementedError()

    def create_assembler(self):
        raise NotImplementedError()

    def _bind_debug_context(self):
        raise NotImplementedError()
