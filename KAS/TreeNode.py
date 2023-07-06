from typing import List, Union, Optional, Dict, Tuple, DefaultDict
import logging
import math
from statistics import mean, stdev
from collections import defaultdict
from functools import partial

from .Node import Path, Node, PseudoNext, AbsolutePath
from .Sampler import Sampler
from .Bindings import Next, Arc
from .Utils import AverageMeter

PseudoTreeNext = Union[Next.Type, int]
PseudoArc = Union[Next.Type, Arc]

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
    def deserialize(serialized: str) -> "TreePath":
        deserialized_list = serialized.split("_")
        return TreePath([Next(Next.Type(int(n[0])), int(n[1:])) for n in deserialized_list])

    @staticmethod
    def decode_next_type(repr_: str):
        if repr_ == "MapReduce":
            return "0"
        elif repr_ == "Shift":
            return "1"
        elif repr_ == "Stride":
            return "2"
        elif repr_ == "Split":
            return "3"
        elif repr_ == "Unfold":
            return "4"
        elif repr_ == "Merge":
            return "5"
        elif repr_ == "Share":
            return "6"
        elif repr_ == "Finalize":
            return "7"

    @ staticmethod
    def decode_str(str_repr: str) -> "TreePath":
        str_repr = str_repr[1:-1].split(", ")
        str_repr = "_".join([TreePath.decode_next_type(
            r[:-1].split("(")[0])+r[:-1].split("(")[1] for r in str_repr])
        return TreePath.deserialize(str_repr)

    def __init__(self, path: List[PseudoTreeNext]) -> None:
        """abs_path records [(op, hash)]"""
        self.abs_path: AbsolutePath = [self.to_next(n) for n in path]

    def __getitem__(self, key):
        return self.abs_path[key]

    def concat(self, next: PseudoTreeNext) -> "TreePath":
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
        suffix = ""
        if len(self.abs_path) > 0 and self.abs_path[-1].key == 0:
            suffix = str(self.abs_path[-1].type)
            full_path = self.abs_path[:-1]
        return sampler.path_to_strs(full_path), suffix if suffix != "" else sampler.path_to_strs(full_path)


class TreeNode:
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
            - state: AverageMeter
            - NumVisitToChild
        2. mid node: a node that represents a type.
            - node
            - is_mid = True
            - type
            - state: AverageMeter
            - NumVisitToChild
        """
        # identifications
        self._node: Node = node
        self._is_mid: bool = is_mid
        self._type: Next.Type = type
        
        # states
        self.state = AverageMeter(support_std=True)
        self._last_T: int = 0
        self._is_dead: bool = False
        self._isin_tree: bool = False
        
        self.l_rave: DefaultDict[PseudoArc, AverageMeter] = defaultdict(AverageMeter)
        
        # conditional members
        if node.is_final():
            self.reward: float = -1
            self.filtered: bool = False
        if self._is_mid:
            self.edge_states: Dict[int, AverageMeter] = defaultdict(partial(AverageMeter, support_std=True))
        else:
            self.children: List["TreeNode"] = []
            primitives = self._node.collect_operations()
            for child in primitives.keys():
                self.children.append(TreeNode(node, is_mid=True, type=child))

    def __eq__(self, __value: "TreeNode") -> bool:
        eq_flag = self._node.__eq__(__value._node) and \
            self._is_mid == __value._is_mid and \
            self._type == __value._type and \
            self.state_dict == __value.state_dict and \
            self.l_rave == __value.l_rave
        if self._node.is_final():
            eq_flag = eq_flag and self.filtered == __value.filtered and self.reward == __value.reward
        if self._is_mid:
            eq_flag = eq_flag and self.edge_states == __value.edge_states
        return eq_flag

    def __hash__(self) -> int:
        return hash((self.to_node(), self._is_mid, self._type))
    
    @property
    def mean(self) -> float:
        return self.state.mean
    
    @property
    def std(self) -> float:
        return self.state.std
    
    @property
    def N(self) -> float:
        return self.state.N
    
    @property
    def state_dict(self) -> Dict:
        return {
            "state": self.state.serialize(),
            "_last_T": self._last_T,
            "_is_dead": self._is_dead,
            "_isin_tree": self._isin_tree
        }
    
    def empty(self) -> bool:
        """
        Whether the node is empty. A subtree consisting of only empty nodes can be safely discarded during garbage collection. 
        """
        empty_flag = self.state.empty() and all([lrave.empty() for lrave in self.l_rave.values()]) and not self._is_dead and self.N == 0 and not self.is_final()
        if self._is_mid:
            empty_flag = empty_flag and all([edge.empty() for edge in self.edge_states.values()])
        return empty_flag
        
    def load(self, state_dict: Dict) -> None:
        self.state.load(state_dict["state"])
        self._last_T = state_dict["_last_T"]
        self._is_dead = state_dict["_is_dead"]
        self._isin_tree = state_dict["_isin_tree"]
    
    def update(self, reward: float, arc: PseudoArc=None) -> None:
        self.state.update(reward)
        if arc and self._is_mid:
            nxt = arc.to_next()
            self.edge_states[nxt.key].update(reward)
    
    def update_edge(self, reward: float, key: int) -> None:
        assert self._is_mid
        self.edge_states[key].update(reward)
    
    def update_lrave(self, reward: float, arc: PseudoArc) -> None:
        assert not self.is_final()
        self.l_rave[arc].update(reward)

    def children_count(self, factory: Dict[Node, "TreeNode"], on_tree: bool=False) -> int:
        """
        Get the number of all children of a node.
        """
        primitives = self._node.collect_operations()
        if self._is_mid:
            nexts = primitives[self._type]
            count = 0
            for next in nexts:
                child = self._node.get_child(Next(self._type, next))
                if child is None:
                    continue
                if on_tree and child in factory and not factory[child]._isin_tree:
                    continue
                if child not in factory or not factory[child].is_dead_end(factory):
                    count += 1
            return count
        else:
            return len(self.children) - sum([child.is_dead_end(factory) or (on_tree and not child._isin_tree) for child in self.children])

    def is_fully_in_tree(self, factory: Dict[Node, "TreeNode"]) -> bool:
        """
        Get all nexts of a node. 
        """
        primitives = self._node.collect_operations()
        if self._is_mid:
            nexts = primitives[self._type]
            for next in nexts:
                child = self._node.get_child(Next(self._type, next))
                if child not in factory or not (factory[child]._isin_tree or factory[child].is_dead_end(factory)):
                    return False
            return True
        else:
            return all([c._isin_tree for _, c, _ in self.get_children(factory)])
    
    def flush_T(self, T:int, factory: Dict[Node, "TreeNode"], g_rave: Dict[Arc, AverageMeter], c_l: float, b: float) -> None:
        if self._last_T == T:
            return
        Tp = math.floor(T ** b)
        orig_Tp = math.floor(self._last_T ** b)
        if Tp > orig_Tp:
            for _ in range(Tp - orig_Tp):
                if not self.is_fully_in_tree(factory):
                    self.add_new_children(factory, g_rave, c_l)
        self._last_T = T
    
    def add_new_children(self, factory: Dict[Node, "TreeNode"], g_rave: Dict[Arc, AverageMeter], c_l: float) -> None:
        """
        Add a new children.
        """
        logging.debug("Add new children to {}".format(self))
        assert not self.is_fully_in_tree(factory)
        def rave(key: Tuple[PseudoTreeNext, TreeNode, AverageMeter]) -> float:
            """
            (1-β) l-RAVE + β g-RAVE
            """
            next, _, _ = key
            if self._is_mid:
                arc = self._node.get_arc_from_handle(Next(self._type, next))
                assert arc is not None
            else:
                arc = next
            beta = c_l / (c_l + self.l_rave[arc].N)
            return (1 - beta) * self.l_rave[arc].mean + beta * g_rave[arc].mean
            
        unadded_children = self.get_unadded_children(factory)
        if len(unadded_children) == 0:
            assert self.is_fully_in_tree(factory), f"{self} is not fully expanded"
            return
        _, child, _ = max(unadded_children, key=rave)
        child._isin_tree = True

    def get_unadded_children(self, factory: Dict[Node, "TreeNode"]) -> List[Tuple[PseudoTreeNext, "TreeNode", AverageMeter]]:
        children = self.get_children(factory)
        unadded_children = [child for child in children if not child[1]._isin_tree]
        return unadded_children
    
    def get_unexpanded_children(self, factory: Dict[Node, "TreeNode"], on_tree: bool=False) -> List[Tuple[PseudoTreeNext, "TreeNode", AverageMeter]]:
        children = self.get_children(factory)
        unexpanded_children = [child for child in children if child[1].N == 0]
        if on_tree: 
            return [child for child in unexpanded_children if child[1]._isin_tree]
        return unexpanded_children
    
    def get_children(self, factory: Dict[Node, "TreeNode"], auto_initialize: bool=True, on_tree: bool=False) -> List[Tuple[PseudoTreeNext, "TreeNode", AverageMeter]]:
        """
        Get all children of a node plus the nexts. Since the tree is searching in the background, we shall get the handles frequently. 
        If some children is dead, we remove them
        """
        primitives = self._node.collect_operations()

        if self._is_mid:
            nexts = primitives[self._type]
            children = [self.get_child(nxt, factory, auto_initialize) for nxt in nexts]
            
            # Remove filtered and dead children. 
            filtered = [
                (nxt, c[0], c[1]) 
                for c, nxt in zip(children, nexts) 
                    if (c is not None) and not c[0].is_dead_end(factory)
            ]
            nexts = [nxt for nxt, _, _ in filtered]
            children = [c for _, c, _ in filtered]
            edge_states = [edge for _, _, edge in filtered]
        else:
            children = self.children
            children = [
                child
                for child in children
                if child._type in primitives.keys() and not child.is_dead_end(factory)
            ]
            nexts = [child._type for child in children]
            edge_states = [child.state for child in children]
            self.children = children
        assert len(nexts) == len(children) == len(edge_states)
        if auto_initialize and not on_tree and len(children) == 0: # No children exists
            self._is_dead = True
        if on_tree:
            return [(nxt, c, e) for nxt, c, e in zip(nexts, children, edge_states) if c._isin_tree]
        return list(zip(nexts, children, edge_states))

    def get_child(self, next: PseudoTreeNext, factory: Dict[Node, "TreeNode"] = None, auto_initialize: bool=True, on_tree: bool=False) -> Optional[Tuple["TreeNode", AverageMeter]]:
        """
        Get the child node of a node with a Next. When the node is dead, return None.
        """
        if self._is_mid:
            assert isinstance(next, int)
            child = self._node.get_child(Next(self._type, next))
            if child is None:
                return None
            if on_tree and (child not in factory or not factory[child]._isin_tree):
                return None
            if child not in factory:
                if auto_initialize:
                    factory[child] = TreeNode(child.to_node())
                else:
                    return None
            return factory[child], self.edge_states[next]
        else:
            assert isinstance(next, Next.Type)
            for _, child, edge_state in self.get_children(factory):
                if child._type == next:
                    return child, edge_state if child._isin_tree else None
            return None

    def is_terminal(self, factory: Dict[Node, "TreeNode"]) -> bool:
        """Check if a node is final, which means it can be realized as a Halide kernel."""
        return self.is_final() or self.is_dead_end(factory)
    
    def is_dead_end(self, factory: Dict[Node, "TreeNode"]) -> bool:
        """Check if a node is final, which means it can be realized as a Halide kernel."""
        if self.is_final() and not self.filtered:
            return False
        if self._node.is_dead_end() or self.is_final():
            self._is_dead = True
        if self._is_dead:
            return True
        
        primitives = self._node.collect_operations()
        if self._is_mid:
            nexts = primitives[self._type]
            dead_children: List[Node] = []
            for next in nexts:
                child = self._node.get_child(Next(self._type, next))
                if child is None:
                    continue
                dead_children.append(child)
                if child not in factory or not factory[child].is_dead_end(factory):
                    return False
            self._is_dead = True
            return True
        else:
            self.get_children(factory)
            if len(self.children) == 0 or all([child.is_dead_end(factory) for child in self.children]):
                self.is_dead = True
                return True
            else:
                return False
    
    def is_final(self) -> bool:
        """
        Check if a node is final, which means it can be realized as a Halide kernel. 
        """
        if self._is_mid:
            return False
        return self._node.is_final()
    
    def to_node(self) -> "Node":
        return self._node.to_node()

    def __repr__(self) -> str:
        if self._is_mid:
            return str(self._node) + "->" + str(self._type).split(".")[-1]
        return str(self._node)
