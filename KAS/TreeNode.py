from typing import List, Union, Optional, Dict, Tuple
import logging

from .Node import Path, Node, PseudoNext, AbsolutePath
from .Sampler import Sampler
from . import Bindings
from .Bindings import Next

PseudoTreeNext = Union[Next.Type, int]


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
        return TreePath([Next(Next.Type(int(n[0])), int(n[1:])) for n in deserialized_list])

    @staticmethod
    def decode_next_type(repr_: str):
        if repr_ == 'MapReduce':
            return '0'
        elif repr_ == "Shift":
            return '1'
        elif repr_ == "Stride":
            return '2'
        elif repr_ == "Split":
            return '3'
        elif repr_ == "Unfold":
            return '4'
        elif repr_ == "Merge":
            return '5'
        elif repr_ == "Share":
            return '6'
        elif repr_ == "Finalize":
            return '7'

    @ staticmethod
    def decode_str(str_repr: str) -> 'TreePath':
        str_repr = str_repr[1:-1].split(', ')
        str_repr = '_'.join([TreePath.decode_next_type(
            r[:-1].split('(')[0])+r[:-1].split('(')[1] for r in str_repr])
        return TreePath.deserialize(str_repr)

    def __init__(self, path: List[PseudoTreeNext]) -> None:
        """abs_path records [(op, hash)]"""
        self.abs_path: AbsolutePath = [self.to_next(n) for n in path]

    def __getitem__(self, key):
        return self.abs_path[key]

    def concat(self, next: PseudoTreeNext) -> 'TreePath':
        if isinstance(next, Next.Type):
            return TreePath(self.abs_path + [Next(next, 0)])
        elif isinstance(next, int):
            return TreePath(self.abs_path[:-1] + [Next(self.abs_path[-1].type, next)])
        else:
            raise Exception("Unexpected next!")
    
    def is_root(self):
        return self.abs_path == []

    def path_to_strs(self, sampler: Sampler):
        full_path = self.abs_path
        suffix = ''
        if len(self.abs_path) > 0 and self.abs_path[-1].key == 0:
            suffix = str(self.abs_path[-1].type)
            full_path = self.abs_path[:-1]
        return sampler.path_to_strs(full_path), suffix if suffix != '' else sampler.path_to_strs(full_path)


class TreeNode(Node):
    """
    A wrapper of node that represents either a type or a full node
    """

    def __init__(self, node: Node, is_mid: bool = False, type: Next.Type = None) -> None:
        """
        node: the underlying Node of this node or of its father (if it is a mid node). 

        Two types of TreeNode for two step search.

        1. leaf node: a node that represents a full node. 
            - node
            - is_mid = False
            - children: Dict[Next, TreeNode]
            - n
            - q
            - NumVisitToChild
        2. mid node: a node that represents a type.
            - node
            - is_mid = True
            - type
            - n
            - q
            - NumVisitToChild
        """
        self._node = node
        self._is_mid: bool = is_mid
        self._type: Next.Type = type
        self.N: int = 0
        self.Q: float = 0
        if self._is_mid and self._type == Next.Type.Finalize:
            self.filtered: List[int] = []
        else:
            self.children: List['TreeNode'] = []

        logging.debug(f"collecting {self._node}")
        assert isinstance(self._node, Node)
        primitives = self._node.collect_operations()
        # Initialize TreeNodes for children.
        if not self._is_mid:
            for child in primitives.keys():
                self.children.append(TreeNode(node, is_mid=True, type=child))
        logging.debug("initialized")

    def __eq__(self, __value: 'TreeNode') -> bool:
        eq_flag = self._node.__eq__(__value._node) and \
            self._is_mid == __value._is_mid and \
            self._type == __value._type and \
            self.N == __value.N and \
            self.Q == __value.Q
        if self._is_mid and self._type == Next.Type.Finalize:
            eq_flag = eq_flag and self.filtered == __value.filtered
        return eq_flag

    def __hash__(self) -> int:
        return hash((self.to_node(), self._is_mid, self._type))

    def get_unexpanded_children(self, factory: Dict[Node, 'TreeNode']) -> List[Tuple[PseudoTreeNext, 'TreeNode']]:
        children = self.get_children(factory)
        unexpanded_children = [child for child in children if child[1].N == 0]
        return unexpanded_children

    def set_child_dead(self, key: int) -> None:
        """
        TOTST
        Set a child Finalize(key) to be dead. I should be a mid node with type Finalize
        """
        assert self._is_mid
        assert self._type == Next.Type.Finalize
        self.filtered.append(key)

    def children_count(self, factory) -> int:
        """Get the number of all children of a node."""
        return len(self.get_children(factory))

    def get_children(self, factory) -> List[Tuple[PseudoTreeNext, 'TreeNode']]:
        """
        Get all children of a node plus the nexts. Since the tree is searching in the background, we shall get the handles frequently. 
        If some children is dead, we remove them
        """
        primitives = self._node.collect_operations()

        # TOTST: remove filtered finalize.
        if self._is_mid:
            nexts = primitives[self._type]
            if self._type == Next.Type.Finalize:
                nexts = [
                    nxt for nxt in nexts if self.get_child(nxt, factory) not in self.filtered and self.get_child(nxt, factory) is not None]
            children = [self.get_child(nxt, factory) for nxt in nexts]
            nexts = [nxt for c, nxt in zip(children, nexts) if c is not None]
            children = [c for c in children if c is not None]
        else:
            children = self.children
            children = [
                child
                for child in children
                if child._type in primitives.keys() and child.children_count(factory) > 0
            ]
            nexts = [child._type for child in children]
            self.children = children
        assert len(nexts) == len(children)

        return list(zip(nexts, children))

    def get_child(self, next: PseudoTreeNext, factory: Dict[Node, 'TreeNode'] = None) -> Optional['TreeNode']:
        """
        Get the child node of a node with a Next. When the node is dead, return None.
        """
        if self._is_mid:
            assert isinstance(next, int)
            child = self._node.get_child(Next(self._type, next))
            if child is None:
                return None
            if child not in factory:
                factory[child] = TreeNode(child.to_node())
            return factory[child]
        else:
            assert isinstance(next, Next.Type)
            for nxt, child in self.get_children(factory):
                if nxt == next:
                    return child
            return None

    def is_final(self) -> bool:
        """Check if a node is final, which means it can be realized as a Halide kernel."""
        if self._is_mid:
            return False
        return self._node.is_final()
    
    def to_node(self) -> 'Node':
        return self._node.to_node()

    def __repr__(self) -> str:
        if self._is_mid:
            return str(self._node) + '->' + str(self._type)
        return str(self._node)
