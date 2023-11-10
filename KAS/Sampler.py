import os
import logging
import torch
from torch import nn
from typing import Any, List, Dict, Optional, Tuple, Union
import random

from . import Bindings
from .Assembler import Assembled, Assembler
from .Bindings import CodeGenOptions
from .KernelPack import KernelPack, KernelLoader
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
            node for node in net.modules() if isinstance(node, Placeholder)
        ]
        return placeholders

    @staticmethod
    def _get_all_mappings(placeholders: List[Placeholder]) -> List[Dict[str, int]]:
        if len(placeholders) == 0:
            logging.warning("No placeholders found in the network.")
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

    def __init__(
        self,
        input_shape: str,
        output_shape: str,
        primary_specs: List[str],
        coefficient_specs: List[str],
        net: nn.Module = None,
        fixed_io_pairs: List[Tuple[int, int]] = [],
        seed: int = 42,
        depth: int = 4,
        max_chain_length: int = 5,
        maximum_tensors: int = 2,
        maximum_reductions: int = 2,
        max_flops: float = 1e15,
        max_rdom_size_multiplier: int = 32,
        enable_flops_based_pruning: bool = True,
        maximum_enumerations_per_var: int = 5,
        maximum_variables_in_size: int = 16,
        maximum_variables_powers_in_size: int = 16,
        requires_exact_division: bool = True,
        requires_odd_kernel_size_in_unfold: bool = True,
        count_coefficients_in_weights_as_allowance_usage: bool = False,
        expression_one_tensor: str = "in_0",
        expression_two_tensors: str = "in_0 * in_1",
        expression_three_tensors: str = "in_0 * in_1 * in_2",
        expression_four_tensors: str = "in_0 * in_1 * in_2 * in_3",
        maximum_finalizations: int = 5,
        allow_weight_permutation: bool = False,
        max_strided_dim_size: int = 30,
        max_unfold_kernel_size: int = 30,
        minimum_unfold_ratio: float = 2.0,
        maximum_valid_reshape_shift_pattern: float = 5.0,
        disallow_merge_input_and_weight: bool = False,
        disallow_tile: bool = True,
        disallow_share_weights: bool = False,
        max_expansion_repeat_multiplier: int = 1,
        max_expansion_merge_multiplier: int = 128,
        max_expansion_weights_sharing_dim_size: int = 3,
        min_expansion_weights_sharing_dim_size: int = 8,
        canonicalize_unfold_order: bool = True,
        maximum_expands: int = -1,
        maximum_merges: int = -1,
        maximum_splits: int = -1,
        maximum_shifts: int = -1,
        maximum_strides: int = -1,
        maximum_unfolds: int = -1,
        maximum_shares: int = -1,
        num_worker_threads: int = 12,
        save_path: str = "./samples",
        halide: bool = False,
        cuda: bool = False,
        autoscheduler: CodeGenOptions.AutoScheduler = CodeGenOptions.AutoScheduler.ComputeRoot,
        extra_options: Dict[str, str] = {},
        rfactor_threshold: int = 32,
        in_bounds_likely_threshold: float = 0.3,
    ):
        """
        Parameters
        ----------
        input_shape : str
            The shape of the desired input tensor. All sizes are represented by variables. A variable can be denoted by a (Python) identifier. For exampler, this argument can be `"[N, C, H, W]"` or `"[N, C, H, W, D]"`. Note the brackets.
        output_shape : str
            Same as above.
        primary_specs : List[str]
            The variables in KAS are of either of the two types: primary, and coefficient. There is no clear distinction, but generally primary variables are large in size (for example, spatial dimensions `H` and `W`), and coefficient variables are small in size (for example, the stride `s`).
            Primary variables are never allowed to be in the denominator of a fraction, while coefficient variables are allowed to be in the denominator. This is utilized to reduce the search space. For example, `s^-1*H` is a legal size.
            This argument is a list of primary variables specifications, along with their (optional) values and (optional) maximum occurrences. For example, the argument `["N: 0", "C", "H = 28: 2", "W = 28: 2"]` means we do not want `N` to be appear in the search space, the spatial dimensions are 28 and can appear at most twice in any phase of searching.
            The syntax of a specification is:
            `SizeSpec` ::= `Size` (`:` `int`)?
            where the integer is the maximum occurrences of a size. And
            `Size` ::= `int` | `id` | `id` `=` `int`
            A size can be anonymous (in which case it is a numeric constant), or named with optional specified value.
        coefficient_specs : List[str]
            Same as above. Note that you can utilize anonymous sizes to write something like `["5", "3: 2"]`. It is recommended that you do not set 2 variables of the exact same size, because this simply doubles the search space. Unless you have a special reason to do so, for example, you need to distinguish between the height `H` and width `W`, which appear in input and output shapes.
        net : nn.Module, optional
            The network to be sampled. This is needed to extract the concrete sizes of each variable from the `Placeholder`s. It is highly recommended that you pass your network in, because this affects the search space!
        fixed_io_pairs : List[Tuple[int, int]], optional
            The dimension pairs in input and output shapes that you want to exclude from the searching process. You can do this to the batch size dimension. For example, if your input and output shapes are both `[N, C, H, W]`, then you can pass `[(0, 0)]` to tell KAS that the first dimension of input and first dimension of output are fixed together.
        seed : int, optional
            The random seed provided to the C++ bindings.
        depth : int, optional
            The maximum number of primitives (excluding FinalizeOp) allowed in a kernel. This is effectively the depth of the search tree, because we add one primitive at one time.
        max_chain_length : int, optional
            The length of each chain of primitives are not allowed to exceed this number.
        maximum_tensors : int, optional
            Maximum number of tensors that this kernel accepts. That is, the maximum number of weights plus 1.
        maximum_reductions : int, optional
            Maximum number of ReduceOp's.
        max_flops : float, optional
            Maximum number of floating point operations allowed in a kernel. This is a soft constraint, because we do not know the exact number of floating point operations in a kernel until we finalize it.
        max_rdom_size_multiplier : int, optional
            We allow for a matmul, times this multiplier at most in ReductionStage.
        enable_flops_based_pruning : bool, optional
            Estimate FLOPs during the search to prune.
        maximum_enumerations_per_var : int, optional
            Maximum enumerations per variable. For example, if setting this to 5, then the power could be chosen from s ** (-2) to s ** 2. 
        maximum_variables_in_size : int, optional
            Maximum number of variables in a size. For example, in `c^_1*H^2*W` has 3 variables.
        maximum_variables_powers_in_size : int, optional
            Maximum number of powers of variables in a size. For example, in `c^_1*H^2*W` has 4.
        requires_exact_division : bool, optional
            Whether to require exact division in each fraction. For example, `H/k` where `H=5`, `k=3` is not allowed if this is set to `True`.
        requires_odd_kernel_size_in_unfold : bool, optional
            Whether to require odd kernel size in UnfoldOp. For example, `Unfold H -> H, s` where `H=5`, `s=2` is not allowed if this is set to `True`.
            Note that this option must be enabled only if `requires_exact_division` is enabled. This is becase, well, you have to make sure a fraction is an integer before deciding whether it is odd or not.
        count_coefficients_in_weights_as_allowance_usage : bool, optional
            Primary variables in Share consume maxOccurrences. But whether coefficient variables are needed to be counted as well can be controlled.
        expression_one_tensor : bool, optional
            The blending operation that blends the input tensors. The ith input tensor is denoted by the identifier `in_i`. Allowed operations are `+`, `*` and `(`, `)`.
            In the case of 1 tensor, this can only be `in_0`.
        expression_two_tensors : bool, optional
            Same as above. Examples are `in_0 + in_1` and `in_0 * in_1`.
        expression_three_tensors : bool, optional
            Same as above. Examples are `in_0 * in_1 + in_2` and `in_0 * in_1 * in_2`.
        expression_four_tensors : bool, optional
            Same as above. Examples are `in_0 * in_1 + in_2 * in_3` and `in_0 * in_1 * in_2 * in_3`.
        maximum_finalizations : int, optional
            Maximum number of FinalizeOp's in a final node. This keeps the top-`k` (`k == maximum_finalizations`) finalizations, which minimize the variance of weights.
        allow_weight_permutation : bool, optional
            Since the analysis of tensor expressions is still not complete, you have to manually tell KAS whether the weights are commutative. If you set this to `True`, then KAS will try all permutations of weights in the tensor expressions. Otherwise, KAS will sort the weights in order of hash.
        max_strided_dim_size : int, optional
            Maximum size of a strided dimension. For example, `Stride s*H -> H` will not be sampled if `s*H > max_strided_dim_size`.
        max_unfold_kernel_size : int, optional
            Maximum size of the parameter of UnfoldOp.
        minimum_unfold_ratio : float, optional
            Minimum ratio of the size of the input dimension to the parameter of UnfoldOp.
        maximum_valid_reshape_shift_pattern : float, optional
            When Shift coincides with a reshape with rather large RHS, this Shift is basically interchangeable with that reshape. For example, if this is set to 10, then a Shift by 1 on a Merge with block size 20 is interchangeable. We put Shift as close to output as possible in this case.
        disallow_merge_input_and_weight : bool, optional
            Merging input tensor and weight tensor via ExpandOp.
        disallow_tile : bool, optional
            Generate ExpandOp above lhs of MergeOp.
        disallow_share_weights : bool, optional
            Use Expand to Share weights.
        max_expansion_repeat_multiplier : int, optional
            Maximum times of expansion in ExpandOp, for repeat.
        max_expansion_merge_multiplier : int, optional
            Maximum times of expansion in ExpandOp, for merge.
        max_expansion_weights_sharing_dim_size : int, optional
            Maximum size of the dim that is shared by 2 weights.
        min_expansion_weights_sharing_dim_size : int, optional
            Minimum size of the dim that is shared by 2 weights. Note that this only restricts the largest set of mappings.
        canonicalize_unfold_order : bool, optional
            Make chained UnfoldOp's appear in ascending parameter order.
        maximum_expands : int, optional
            Maximum number of ExpandOp's. `-1` for infinite.
        maximum_merges : int, optional
            Maximum number of MergeOp's. `-1` for infinite.
        maximum_splits : int, optional
            Maximum number of SplitOp's. `-1` for infinite.
        maximum_shifts : int, optional
            Maximum number of ShiftOp's. `-1` for infinite.
        maximum_strides : int, optional
            Maximum number of StrideOp's. `-1` for infinite.
        maximum_unfolds : int, optional
            Maximum number of UnfoldOp's. `-1` for infinite.
        maximum_shares : int, optional
            Maximum number of ShareOp's. `-1` for infinite.
        num_worker_threads : int, optional
            Number of threads for parallel search and expansion.
        save_path : str, optional
            The path to save the sampled kernels.
        halide : bool, optional
            Generate Halide pipelines.
        cuda : bool, optional
            Use GPU.
        autoscheduler : Bindings.CodeGenOptions.AutoScheduler, optional
            Halide autoschdulers. Check `src/Python/Bindings.cpp` for detail.
        extra_options : Dict[str, str], optional
            Extra options for Halide autoschedulers. Note that all the parameters are passed in as strings, as required by Halide.
            Specifically for Anderson2021, there are:
            - `parallelism` (default "16"): Maximum level of parallelism available. (i.e., number of SMs.)
            - `beam_size` (default "32"): Beam size to use in the beam search. Defaults to 32. Use 1 to get a greedy search instead.
            - `random_dropout` (default "100"): Percent chance of accepting each state in the beam. Normalized by the number of decisions made, so 5 would be there's a 5 percent chance of never rejecting any states.
            - `search_space_options` (default "1111"): Expects a string of four 0/1 values that allow/disallow the following options:
                compute root, inline, compute at the block level, compute at the thread level
              e.g. 1000 would allow compute root only
            - `num_passes` (default "0"): User-requested specific number of passes. Ignored if 0.
            - `shared_memory_limit_kb` (default "48"): Shared memory limit per block for the target GPU.
            - `shared_memory_sm_limit_kb` (default "96"): Shared memory limit per SM for the target GPU.
            - `active_block_limit` (default "32"): Maximum number of active blocks for the target GPU.
            - `active_warp_limit` (default "64"): Maximum number of active warps for the target GPU.
        rfactor_threshold : int, optional
            Halide autoschedulers are weak, so we manually split up the reduction loops for them. If the outermost loop is smaller than this parameter, we will not perform manual splitting of reduction loops.
        in_bounds_likely_threshold : float, optional
            We need zero padding in UnfoldOp. We use a special Halide directive `likely` to hint that the most of the results are not 0. But sometimes this is not the case, for example, when `H=8` and `k=5`, `Unfold H -> H, k` has a lot of zeros.In this case, if `5/8=0.625 > in_bounds_likely_threshold`, we will not use `likely`.
        """
        options = Bindings.SampleOptions(
            seed=seed,
            depth=depth,
            max_chain_length=max_chain_length,
            maximum_tensors=maximum_tensors,
            maximum_reductions=maximum_reductions,
            max_flops=max_flops,
            max_rdom_size_multiplier=max_rdom_size_multiplier,
            enable_flops_based_pruning=enable_flops_based_pruning,
            maximum_enumerations_per_var=maximum_enumerations_per_var,
            maximum_variables_in_size=maximum_variables_in_size,
            maximum_variables_powers_in_size=maximum_variables_powers_in_size,
            requires_exact_division=requires_exact_division,
            requires_odd_kernel_size_in_unfold=requires_odd_kernel_size_in_unfold,
            count_coefficients_in_weights_as_allowance_usage=count_coefficients_in_weights_as_allowance_usage,
            expression_one_tensor=expression_one_tensor,
            expression_two_tensors=expression_two_tensors,
            expression_three_tensors=expression_three_tensors,
            expression_four_tensors=expression_four_tensors,
            maximum_finalizations=maximum_finalizations,
            allow_weight_permutation=allow_weight_permutation,
            max_strided_dim_size=max_strided_dim_size,
            max_unfold_kernel_size=max_unfold_kernel_size,
            minimum_unfold_ratio=minimum_unfold_ratio,
            maximum_valid_reshape_shift_pattern=maximum_valid_reshape_shift_pattern,
            disallow_merge_input_and_weight=disallow_merge_input_and_weight,
            disallow_tile=disallow_tile,
            disallow_share_weights=disallow_share_weights,
            max_expansion_repeat_multiplier=max_expansion_repeat_multiplier,
            max_expansion_merge_multiplier=max_expansion_merge_multiplier,
            max_expansion_weights_sharing_dim_size=max_expansion_weights_sharing_dim_size,
            min_expansion_weights_sharing_dim_size=min_expansion_weights_sharing_dim_size,
            canonicalize_unfold_order=canonicalize_unfold_order,
            maximum_expands=maximum_expands,
            maximum_merges=maximum_merges,
            maximum_splits=maximum_splits,
            maximum_shifts=maximum_shifts,
            maximum_strides=maximum_strides,
            maximum_unfolds=maximum_unfolds,
            maximum_shares=maximum_shares,
        )
        self._seed = seed
        self._save_path = save_path

        all_mappings = [{
            "N": 4, 
            "C_in": 4, 
            "C_out": 8, 
            "H": 8,
            "W": 8
        }]
        if net is not None:
            all_mappings = Sampler._extract_all_mappings(net)

        self._sampler = Bindings.Sampler(
            input_shape,
            output_shape,
            primary_specs,
            coefficient_specs,
            all_mappings,
            fixed_io_pairs,
            options,
            num_worker_threads,
        )
        self._codegen_options = Bindings.CodeGenOptions(
            halide=halide,
            use_gpu=cuda,
            auto_scheduler=autoscheduler,
            extra_options=extra_options,
            rfactor_threshold=rfactor_threshold,
            in_bounds_likely_threshold=in_bounds_likely_threshold,
        )
        self._device = torch.device("cuda" if cuda else "cpu")

    def get_all_stats(self) -> str:
        """Get all stats."""
        return self._sampler.get_all_stats()

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

    def random_final_nodes_with_prefix(
        self, prefix: PseudoPath, count: int, type: Optional[Next.Type] = None, steps: int = 1
    ) -> List[VisitedNode]:
        """Find final nodes with specified prefix. Note that the returned list may not contain as many nodes as required, and even an empty list can be returned."""
        prefix = Path(prefix)
        visited_nodes = self._sampler.random_final_nodes_with_prefix(
            prefix.abs_path, count, type, steps
        )
        return [VisitedNode(Path(path), node) for path, node in visited_nodes]

    def path_to_strs(self, path: PseudoPath) -> List[str]:
        node = self.root()
        strs = []
        for next in path:
            strs.append(node.get_child_description(next))
            node = node.get_child(next)
        return strs

    def _realize(
        self,
        node: Union[Node, Assembled],
        all_mappings: List[Dict[str, int]],
        name: str,
    ) -> Bindings.Kernel:
        if isinstance(node, Assembled):
            dir_name = f"kernel_manual_{node._name}" if name is None else name
            kernel_name = "kernel_manual"
            return node._realize(
                all_mappings,
                self._codegen_options,
                os.path.join(self._save_path, dir_name),
                kernel_name,
            )
        else:
            dir_name = (
                f"kernel_generated_{abs(hash(node.to_node()))}"
                if name is None
                else name
            )
            kernel_name = "kernel_generated"
            return node._realize_as_final(
                all_mappings,
                self._codegen_options,
                os.path.join(self._save_path, dir_name),
                kernel_name,
            )

    # Note: we need the name to be unique, because it is used for identifying a kernel.
    # If we use the same name for different kernels, the later kernels will not be compiled.
    def realize(
        self, net: nn.Module, node: Union[Node, Assembled], name: str = None
    ) -> KernelLoader:
        placeholders = Sampler._extract_placeholders(net)
        all_mappings = Sampler._get_all_mappings(placeholders)
        return KernelLoader(self._realize(node, all_mappings, name))

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
        seed: int = 0xDEADBEAF,
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
            MockNodeMetadata(
                self, id, **(v if isinstance(v, dict) else {"name": str(v)})
            )
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

        # Assign a path for each node.
        self.visit([]).mock_set_path(Path([]))
        for _ in range(len(self._mock_nodes)):
            # The following loop progresses by at least 1 per iteration.
            for node in self._mock_nodes:
                path = node.mock_get_path()
                if path is not None:
                    for next, child in self._mock_edges[node.mock_get_id()].items():
                        if child.mock_get_path() is None:
                            child.mock_set_path(path.concat(next))
        for node in self._mock_nodes:
            assert node.mock_get_path() is not None

    def mock_get_children(self, id: int) -> Dict[Next, MockNodeMetadata]:
        return self._mock_edges[id]

    def visit(self, path: PseudoPath) -> Optional[MockVisitedNode]:
        path = Path(path)
        node = self._mock_nodes[0]
        for next in path:
            children = self._mock_edges[node.mock_get_id()]
            if next not in children:
                return None
            node = children[next]
        return MockVisitedNode(path, node)

    def random_node_with_prefix(self, prefix: PseudoPath) -> Optional[VisitedNode]:
        raise NotImplementedError()

    def random_final_nodes_with_prefix(
        self, prefix: PseudoPath, count: int, type: Optional[Next.Type] = None, steps: int = 1
    ) -> List[VisitedNode]:
        node = self.visit(prefix)
        if node is None:
            return []
        if node.is_final():
            return [node]
        
        def random_child(node: MockVisitedNode, type: Optional[Next.Type]=None) -> Optional[MockVisitedNode]:
            # print(f"Random from {node}")
            if node.is_final():
                return node
            
            children = self._mock_edges[node.mock_get_id()]
            if type is not None:
                children = {next: child for next, child in children.items() if next.type == type}
            if len(children) == 0:
                return None
            next = random.choice(list(children.keys()))
            node = MockVisitedNode(node.mock_get_path().concat(next), children[next])
            return random_child(node)
        
        res = []
        for _ in range(count):
            final_node = random_child(node, type=type)
            if final_node is None:
                break
            res.append(final_node)
        
        return res

    def _realize(
        self, node: Union[Node, Assembled], all_mappings: List[Dict[str, int]]
    ) -> Bindings.Kernel:
        raise NotImplementedError()

    def realize(
        self, net: nn.Module, node: Union[Node, Assembled], identifier_prefix: str
    ) -> Tuple[List[KernelPack], int]:
        raise NotImplementedError()

    def create_assembler(self):
        raise NotImplementedError()

    def _bind_debug_context(self):
        raise NotImplementedError()
