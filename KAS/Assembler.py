from typing import Dict, List, Tuple

from . import Bindings


ForwardDimension = Bindings.ForwardDimension

class Assembled:
    def __init__(self, assembler: Bindings.Assembler, tensors: List[List[ForwardDimension]], blending: str):
        self._assembler = assembler
        self._tensors = tensors
        self._blending = blending

    def convert_to_path(self, sampler):
        """Convert this assembled kernel to a path."""
        from .Node import Path
        return Path(self._assembler.convert_assembled_to_path(self._tensors, sampler._sampler))

    def _realize(self, all_mappings: List[Dict[str, int]], halide_options: Bindings.CodeGenOptions) -> Bindings.Kernel:
        return self._assembler.build(self._tensors, self._blending, all_mappings, halide_options)

class Assembler:
    def __init__(self, assembler: Bindings.Assembler):
        self._assembler = assembler

    def get_sizes(self, *names) -> List[Bindings.Size]:
        """Get sizes by their names.
        Example:
        >>> N, C, H, W = assembler.get_sizes('N', 'C', 'H', 'W')
        """
        return self._assembler.get_sizes(list(names))

    def make_dims_of_sizes(self, *names) -> List[ForwardDimension]:
        """Make input dimensions (they come from tensors) from sizes.
        Example:
        >>> N, C, H, W = assembler.get_sizes('N', 'C', 'H', 'W')
        >>> dN, dHW = assembler.make_dims_of_sizes(N, H * W)
        """
        return self._assembler.make_dims_of_sizes(list(names))

    def create_merge(self, lhs: ForwardDimension, rhs: ForwardDimension) -> ForwardDimension:
        """Create a MergeOp."""
        return self._assembler.create_merge(lhs, rhs)

    def create_share(self, lhs: ForwardDimension, rhs: ForwardDimension) -> ForwardDimension:
        """Create a ShareOp."""
        return self._assembler.create_share(lhs, rhs)

    def create_shift(self, input: ForwardDimension, shift: int) -> ForwardDimension:
        """Create a ShiftOp. Note that the shift is not a Size but a constant."""
        return self._assembler.create_shift(input, shift)

    def create_split(self, input: ForwardDimension) -> Tuple[ForwardDimension, ForwardDimension]:
        """Create a SplitOp."""
        return self._assembler.create_split(input)

    def create_stride(self, input: ForwardDimension, stride: Bindings.Size) -> ForwardDimension:
        """Create a StrideOp."""
        return self._assembler.create_stride(input, stride)

    def create_unfold(self, input: ForwardDimension, window: Bindings.Size) -> Tuple[ForwardDimension, ForwardDimension]:
        """Create an UnfoldOp."""
        return self._assembler.create_unfold(input, window)

    def assemble(self, blending: str, *tensors: List[ForwardDimension]) -> Assembled:
        """Build the kernel, with the specified Dimensions as the dimensions in input tensors.
        :param str blending: The expression. For example, `'in_0 * in_1'`.
        """
        assert isinstance(blending, str)
        return Assembled(self._assembler, list(tensors), blending)
