from collections import defaultdict
from typing import List, Union

from .Node import Path, Node, PseudoNext, AbsolutePath
from .Sampler import Sampler
from . import Bindings
from .Bindings import Next

PseudoTreeNext = Union[PseudoNext, Next.Type]


class TreePath(Path):
    """
    A path that decouples the type and content of a node.
    Type: Next.Type
    Content: Hash value (currently) or Parameters 
    """

    def to_next(self, tup: PseudoTreeNext) -> Next:
        if isinstance(tup, Next.Type):
            return Next(tup, 0)
        return super().to_next(tup)

    @ staticmethod
    def deserialize(serialized: str) -> 'TreePath':
        deserialized_list = serialized.split('_')
        return TreePath([Next(Next.Type(n[0]), int(n[1:])) for n in deserialized_list])

    def __init__(self, path: List[PseudoTreeNext]) -> None:
        """abs_path records [(op, hash)]"""
        self.abs_path: AbsolutePath = [self.to_next(n) for n in path]

    def __getitem__(self, key):
        return self.abs_path[key]

    def append(self, next: PseudoTreeNext):
        self.abs_path.append(self.to_next(next))

    def concat(self, next: PseudoTreeNext) -> 'TreePath':
        if isinstance(next, Next.Type):
            return TreePath(self.abs_path + [self.to_next(next)])
        else:
            return TreePath(self.abs_path[:-1] + [Next(self.abs_path[-1].type, next)])

    def path_to_strs(self, sampler: Sampler):
        full_path = self.abs_path
        suffix = ''
        if len(self.abs_path) > 0 and self.abs_path[-1].key == 0:
            suffix = str(self.abs_path[-1].type)
            full_path = self.abs_path[:-1]
        return sampler.path_to_strs(full_path), suffix if suffix != '' else sampler.path_to_strs(full_path)


class TreeNode(Node):
    """
    A node that represents either a type or a full node
    """

    def __init__(self, path: Path, node: Bindings.Node, is_mid: bool = False, type: Next.Type = None) -> None:
        """
        node: the underlying node of this node or of its father (if it is a mid node). 
        """
        super().__init__(node)
        self.path = TreePath(path)
        self._is_mid = is_mid
        self._type = type

        handles = super().get_children_handles()
        primitives = defaultdict(list)
        for handle in handles:
            primitives[handle.type].append(handle.key)

        if is_mid:  # calculate the children upon initialization
            assert type is not None, "Please pass a type for mid nodes."
            self.children = primitives[type]
        else:
            self.children = list(primitives.keys())

    def __eq__(self, __value: object) -> bool:
        return \
            super().__eq__(__value) and \
            self._is_mid == __value._is_mid and \
            self._type == __value._type

    def __hash__(self) -> int:
        return hash(self._node) + hash(self._is_mid) + hash(self._type)

    def children_count(self) -> int:
        """Get the number of all children of a node."""
        return len(self.children)

    def get_children_handles(self) -> List[int]:
        """Get all children of a node."""
        return self.children

    def get_child(self, next: PseudoTreeNext) -> 'TreeNode':
        """Get the child node of a node with a Next."""
        new_path = self.path.concat(next)
        if self._is_mid:
            assert not isinstance(next, Next.Type)
            return TreeNode(new_path, self._node.get_child(new_path[-1]))
        else:
            assert isinstance(next, Next.Type)
            return TreeNode(new_path, self._node, is_mid=True, type=next)

    def is_final(self) -> bool:
        """Check if a node is final, which means it can be realized as a Halide kernel."""
        if self._is_mid:
            return False
        return self._node.is_final()

    def __repr__(self) -> str:
        if self._is_mid:
            return str(self._node) + '->' + str(self._type)
        return str(self._node)
