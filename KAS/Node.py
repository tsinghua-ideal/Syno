from collections import defaultdict
from typing import Dict, List, Tuple, Union

import kas_cpp_bindings
from kas_cpp_bindings import Next


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

    def __init__(self, path: Path, node: kas_cpp_bindings.Node) -> None:
        self.path = path
        self._node = node

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Node):
            return False
        return self.path == __value.path
    
    def __hash__(self) -> int:
        return hash(tuple(self.path))

    def children_count(self) -> int:
        return self._node.children_count()

    def get_children_handles(self) -> List[Next]:
        return self._node.get_children_handles()

    def get_children_types(self) -> Dict[str, int]:
        handles = self.get_children_handles()
        result = defaultdict(int)
        for handle in handles:
            result[str(handle.type)] += 1
        return result

    def get_child(self, next: PseudoNext) -> 'Node':
        return Node(self.path.concat(next), self._node.get_child(Path.to_next(next)))

    def is_final(self) -> bool:
        return self._node.is_final()
    
    def is_dead_end(self) -> bool:
        (not self.is_final()) and self.children_count() == 0
    
    def is_terminal(self) -> bool:
        # Either a final node, or a dead end.
        return self.is_final() or self.children_count() == 0

    def _realize_as_final(self, all_mappings: List[Dict[str, int]], halide_options: kas_cpp_bindings.CodeGenOptions) -> kas_cpp_bindings.Kernel:
        return self._node.realize_as_final(all_mappings, halide_options)

    def __repr__(self) -> str:
        return str(self._node)
