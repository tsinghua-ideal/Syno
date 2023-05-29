from collections import defaultdict
from typing import Dict, List, Tuple, Union

from . import Bindings
from .Bindings import Next


AbsolutePath = List[Next]
'''
AbsolutePath is a list of Next(type=Next.Type, key=int).
'''

PseudoNext = Union[Next, Tuple[str, int]]


class Path:
    """A path in Python, not necessarily corresponding to a C++ path."""

    @staticmethod
    def to_next(tup: PseudoNext) -> Next:
        if isinstance(tup, Next):
            return tup
        t, k = tup
        return Next(getattr(Next, t), k)

    def serialize(self) -> str:
        serialized = [str(int(n.type)) + str(n.key) for n in self.abs_path]
        return '_'.join(serialized)

    @ staticmethod
    def deserialize(serialized: str) -> 'Path':
        deserialized_list = serialized.split('_')
        return Path([Next(Next.Type(n[0]), int(n[1:])) for n in deserialized_list])

    def __init__(self, path: List[PseudoNext]) -> None:
        self.abs_path: AbsolutePath = [Path.to_next(n) for n in path]

    def __len__(self) -> int:
        return len(self.abs_path)

    def __iter__(self):
        return iter(self.abs_path)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Path):
            return False
        return self.abs_path == __value.abs_path

    def __hash__(self) -> int:
        return hash(tuple(self.abs_path))

    def append(self, next: PseudoNext):
        self.abs_path.append(Path.to_next(next))

    def concat(self, next: PseudoNext) -> 'Path':
        return Path(self.abs_path + [Path.to_next(next)])

    def to_identifier(self) -> str:
        return '_'.join(str(next) for next in self.abs_path).replace('(', '').replace(')', '')

    def __repr__(self) -> str:
        return f'[{", ".join(str(next) for next in self.abs_path)}]'


class Node:
    """A node in Python, not necessarily corresponding to a C++ node."""

    def __init__(self, node: Bindings.Node) -> None:
        self._node = node

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Node):
            return False
        return self._node == __value._node

    def __hash__(self) -> int:
        return hash(self._node)

    def children_count(self) -> int:
        """Get the number of all children of a node."""
        return self._node.children_count()

    def get_children_handles(self) -> List[Next]:
        """Get all children of a node."""
        return self._node.get_children_handles()

    def collect_operations(self) -> Dict[Next.Type, List[int]]:
        """Group the children of a node by their type."""
        handles = self.get_children_handles()
        result = defaultdict(list)
        for handle in handles:
            result[handle.type].append(handle.key)
        return result

    def get_children_types(self) -> Dict[str, int]:
        """Count the number of children of each type."""
        handles = self.get_children_handles()
        result = defaultdict(int)
        for handle in handles:
            result[str(handle.type)] += 1
        return result

    def get_child(self, next: PseudoNext) -> 'Node':
        """Get the child node of a node with a Next."""
        return Node(self._node.get_child(Path.to_next(next)))

    def get_child_description(self, next: PseudoNext) -> str:
        """Get the description of Next."""
        return Path.to_next(next).description(self._node)

    def is_final(self) -> bool:
        """Check if a node is final, which means it can be realized as a Halide kernel."""
        return self._node.is_final()

    def is_dead_end(self) -> bool:
        """Check if a node is a dead end, which means it has no children and is not final."""
        return (not self.is_final()) and self.children_count() == 0

    def is_terminal(self) -> bool:
        """Check if a node is terminal, which means it is either final or a dead end."""
        # Either a final node, or a dead end.
        return self.is_final() or self.children_count() == 0

    def _realize_as_final(self, all_mappings: List[Dict[str, int]], halide_options: Bindings.CodeGenOptions) -> Bindings.Kernel:
        return self._node.realize_as_final(all_mappings, halide_options)

    def estimate_total_flops_as_final(self) -> int:
        return self._node.estimate_total_flops_as_final()

    def to_node(self) -> 'Node':
        return Node(self._node)

    def __repr__(self) -> str:
        return str(self._node)

class VisitedNode(Node):
    """Node with Path."""

    def __init__(self, path: Path, node: Bindings.Node) -> None:
        super().__init__(node)
        self.path = path

    def __eq__(self, __value: object) -> bool:
        raise ValueError("VisitedNode should not be compared.")
    def __hash__(self) -> int:
        raise ValueError("VisitedNode should not be hashed.")

    def get_child(self, next: PseudoNext) -> 'VisitedNode':
        """Get the child node of a node with a Next."""
        return VisitedNode(self.path.concat(next), self._node.get_child(Path.to_next(next)))

    def __repr__(self) -> str:
        return f"VisitedNode({self.path}, {self._node})"
