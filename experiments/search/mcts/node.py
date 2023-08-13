import logging
import math
from collections import defaultdict
from functools import partial
from typing import List, Union, Optional, Dict, Tuple, DefaultDict

from KAS.Node import Path, Node, AbsolutePath
from KAS.Sampler import Sampler
from KAS.Bindings import Next, Arc

from .avg_meter import AverageMeter

PseudoTreeNext = Union[Next.Type, int]
PseudoArc = Union[Next.Type, Arc]

dimensions_type = [
    "MapReduce",
    "Expand",
    "Shift",
    "Stride",
    "Split",
    "Unfold",
    "Merge",
    "Share",
    "Finalize",
]


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

    @staticmethod
    def deserialize(serialized: str) -> "TreePath":
        deserialized_list = serialized.split("_")
        return TreePath(
            [Next(Next.Type(int(n[0])), int(n[1:])) for n in deserialized_list]
        )

    @staticmethod
    def decode_next_type(repr_: str):
        try:
            pos = dimensions_type.index(repr_)
        except ValueError:
            raise Exception(f"Unexpected type {repr_}! ")
        return str(pos)

    @staticmethod
    def decode_str(str_repr: str) -> "TreePath":
        str_repr = str_repr[1:-1].split(", ")
        str_repr = "_".join(
            [
                TreePath.decode_next_type(r[:-1].split("(")[0]) + r[:-1].split("(")[1]
                for r in str_repr
            ]
        )
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
        return sampler.path_to_strs(
            full_path
        ), suffix if suffix != "" else sampler.path_to_strs(full_path)


class TreeNode:
    """
    A wrapper of node that represents either a type or a full node
    """

    def __init__(
        self, node: Node, is_mid: bool = False, type: Next.Type = None
    ) -> None:
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
        assert isinstance(node, Node)
        self._node: Node = node
        self._is_mid: bool = is_mid
        self._type: Next.Type = type

        # states
        self.state = AverageMeter(support_std=True)
        self._last_T: int = 0
        self._is_dead: bool = False
        self._exhausted: bool = False
        self._isin_tree: bool = False

        self.l_rave: DefaultDict[PseudoArc, AverageMeter] = defaultdict(AverageMeter)

        # conditional members
        if node.is_final():
            self.reward: float = -1
            self.filtered: bool = False
        if self._is_mid:
            self.edge_states: DefaultDict[int, AverageMeter] = defaultdict(
                partial(AverageMeter, support_std=True)
            )
        else:
            self.children: List["TreeNode"] = []
            primitives = self._node.collect_operations()
            for child in primitives.keys():
                self.children.append(TreeNode(node, is_mid=True, type=child))

    def eq_verb(self, __value: "TreeNode"):
        eq_flag = (
            self._node.__eq__(__value._node)
            and self._is_mid == __value._is_mid
            and self._type == __value._type
            and self.state_dict == __value.state_dict
            and self.l_rave == __value.l_rave
        )
        print(self._node.__eq__(__value._node))
        print(self._is_mid == __value._is_mid)
        print(self._type == __value._type)
        print(self.state_dict == __value.state_dict)
        print(self.l_rave == __value.l_rave)
        if self._node.is_final():
            eq_flag = (
                eq_flag
                and self.filtered == __value.filtered
                and self.reward == __value.reward
            )
            print(self.filtered == __value.filtered)
            print(self.reward == __value.reward)
        if self._is_mid:
            eq_flag = eq_flag and self.edge_states == __value.edge_states
            print(self.edge_states == __value.edge_states)
        return eq_flag

    def __eq__(self, __value: "TreeNode") -> bool:
        eq_flag = (
            self._node.__eq__(__value._node)
            and self._is_mid == __value._is_mid
            and self._type == __value._type
            and self.state_dict == __value.state_dict
            and self.l_rave == __value.l_rave
        )
        if self._node.is_final():
            eq_flag = (
                eq_flag
                and self.filtered == __value.filtered
                and self.reward == __value.reward
            )
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
            "_exhausted": self._exhausted,
            "_isin_tree": self._isin_tree,
        }

    def empty(self) -> bool:
        """
        Whether the node is empty. A subtree consisting of only empty nodes can be safely discarded during garbage collection.
        """
        empty_flag = (
            self.state.empty()
            and all([lrave.empty() for lrave in self.l_rave.values()])
            and not self._is_dead
            and self.N == 0
            and not self.is_final()
        )
        if self._is_mid:
            empty_flag = empty_flag and all(
                [edge.empty() for edge in self.edge_states.values()]
            )
        return empty_flag

    def load(self, state_dict: Dict) -> None:
        self.state.refresh(state_dict["state"])
        self._last_T = state_dict["_last_T"]
        self._is_dead = state_dict["_is_dead"]
        self._exhausted = state_dict["_exhausted"]
        self._isin_tree = state_dict["_isin_tree"]

    ######################### Update states ##########################

    def update(self, reward: float, arc: Optional[Arc] = None) -> None:
        self.state.update(reward)
        if arc:
            self.update_edge(reward, arc.to_next().key)

    def update_edge(self, reward: float, key: int) -> None:
        assert self._is_mid  # only mid node has stored edge states
        self.edge_states[key].update(reward)

    def update_lrave(self, reward: float, arc: PseudoArc) -> None:
        assert not self.is_final()
        self.l_rave[arc].update(reward)

    # Cleansing
    def clear_lrave(self) -> None:
        for k, rave_score in list(self.l_rave.items()):
            if rave_score.empty():
                self.l_rave.pop(k)

    def clear_edge(self) -> None:
        if self._is_mid:
            for k, edge in list(self.edge_states.items()):
                if edge.empty():
                    self.edge_states.pop(k)

    def flush_T(
        self,
        T: int,
        factory: Dict[Node, "TreeNode"],
        g_rave: Dict[Arc, AverageMeter],
        c_l: float,
        b: float,
        filter_exhausted: bool = False,
    ) -> None:
        """
        Flush T (for progressive widening).
        Dependencies:
            children_count -> exhausted -> is_dead_end -> is_final
            is_fully_in_tree -> is_dead_end -> is_final
            reveal_new_children -> get_unrevealed_children -> get_children -> get_child -> is_dead_end -> is_final
        """
        Tp = math.floor(T**b)
        while Tp > self.children_count(
            factory, on_tree=True, filter_exhausted=filter_exhausted
        ):
            if self.is_fully_in_tree(factory):
                break
            else:
                self.reveal_new_children(factory, g_rave, c_l)
        self._last_T = T

    def children_count(
        self,
        factory: Dict[Node, "TreeNode"],
        on_tree: bool = False,
        filter_exhausted: bool = False,
    ) -> int:
        """
        Get the number of all children of a node.
        Dependencies: is_exhausted -> is_dead_end -> is_final -> None
        """
        if self._is_mid:
            nexts = [
                handle.key
                for handle in self._node.get_children_handles()
                if handle.type == self._type
            ]
            count = 0
            for next in nexts:
                child = self._node.get_child(Next(self._type, next))
                if child is None:
                    continue
                if on_tree and child in factory and not factory[child]._isin_tree:
                    continue
                if filter_exhausted:
                    if child not in factory or not factory[child].is_exhausted(factory):
                        count += 1
                else:
                    if child not in factory or not factory[child].is_dead_end(factory):
                        count += 1
            return count
        else:
            if filter_exhausted:
                return len(self.children) - sum(
                    [
                        child.is_exhausted(factory)
                        or (on_tree and not child._isin_tree)
                        for child in self.children
                    ]
                )
            else:
                return sum(
                    [
                        not child.is_dead_end(factory)
                        and not (on_tree and not child._isin_tree)
                        for child in self.children
                    ]
                )

    def reveal_new_children(
        self,
        factory: Dict[Node, "TreeNode"],
        g_rave: Dict[Arc, AverageMeter],
        c_l: float,
    ) -> bool:
        """
        Reveal a new children to its parent.
        Return success or not.

        Dependencies: get_unrevealed_children -> get_children -> get_child -> is_dead_end -> is_final -> None
        """

        def rave(key: Tuple[PseudoTreeNext, TreeNode, AverageMeter]) -> float:
            """
            (1-β) l-RAVE + β g-RAVE
            """
            next, _, _ = key
            if self._is_mid:
                assert isinstance(next, int)
                arc = self._node.get_arc_from_handle(Next(self._type, next))
                if arc is None:  # the next is already dead
                    return -1
            else:
                assert isinstance(next, Next.Type)
                arc = next
            beta = c_l / (c_l + self.l_rave[arc].N)
            return (1 - beta) * self.l_rave[arc].mean + beta * g_rave[arc].mean

        unrevealed_children = self.get_unrevealed_children(factory)
        if len(unrevealed_children) == 0:
            # logging.debug("No new children to be added")
            assert self.is_fully_in_tree(
                factory
            ), f"{self} is not fully expanded but no children can be revealed"
            return False
        key = max(unrevealed_children, key=rave)
        if rave(key) == -1:
            # logging.debug("No new children to be added")
            return False
        _, child, _ = key
        child._isin_tree = True
        return True

    def get_unrevealed_children(
        self, factory: Dict[Node, "TreeNode"]
    ) -> List[Tuple[PseudoTreeNext, "TreeNode", AverageMeter]]:
        """
        Get all unrevealed children of a node with nexts and edge states.
        Dependencies: get_children -> get_child -> is_dead_end -> is_final -> None
        """
        children = self.get_children(factory)
        unrevealed_children = [child for child in children if not child[1]._isin_tree]
        return unrevealed_children

    def get_unexpanded_children(
        self, factory: Dict[Node, "TreeNode"], on_tree: bool = False
    ) -> List[Tuple[PseudoTreeNext, "TreeNode", AverageMeter]]:
        """
        Get all unexpanded children of a node with nexts and edge states.
        Dependencies: get_children -> get_child -> is_dead_end -> is_final -> None
        """
        children = self.get_children(factory, on_tree=on_tree)
        unexpanded_children = [child for child in children if child[1].N == 0]
        return unexpanded_children

    def get_children_nexts(
        self, factory: Dict[Node, "TreeNode"]
    ) -> List[PseudoTreeNext]:
        """
        Get nexts to all alive children without initialize them.
        Dependencies: get_child -> is_dead_end -> is_final -> None
        """
        if self._is_mid:
            nexts = [
                handle.key
                for handle in self._node.get_children_handles()
                if handle.type == self._type
            ]
            children = [self._node.get_child(Next(self._type, nxt)) for nxt in nexts]
            nexts = [
                nxt
                for nxt, ch in zip(nexts, children)
                if ch is not None
                and (not ch in factory or not factory[ch].is_dead_end(factory))
            ]
        else:
            nexts = [
                child._type for child in self.children if not child.is_dead_end(factory)
            ]

        return nexts

    def get_children(
        self,
        factory: Dict[Node, "TreeNode"],
        auto_initialize: bool = True,
        on_tree: bool = False,
    ) -> List[Tuple[PseudoTreeNext, "TreeNode", AverageMeter]]:
        """
        Get all alive children of a node with nexts and edge states.
        Dependencies: get_child -> is_dead_end -> is_final -> None
        """
        if on_tree:
            auto_initialize = False

        if self._is_mid:
            nexts = [
                handle.key
                for handle in self._node.get_children_handles()
                if handle.type == self._type
            ]
            children = [self.get_child(nxt, factory, auto_initialize) for nxt in nexts]

            # Remove filtered and dead children.
            filtered = [
                (nxt, c[0], c[1]) for c, nxt in zip(children, nexts) if c is not None
            ]
            nexts = [nxt for nxt, _, _ in filtered]
            children = [c for _, c, _ in filtered]
            edge_states = [edge for _, _, edge in filtered]
        else:
            children = self.children
            children = [child for child in children if not child.is_dead_end(factory)]
            nexts = [child._type for child in children]
            edge_states = [child.state for child in children]
            self.children = children

        assert len(nexts) == len(children) == len(edge_states)
        ret_list = list(zip(nexts, children, edge_states))
        if on_tree:
            ret_list = [(nxt, c, e) for nxt, c, e in ret_list if c._isin_tree]
        return ret_list

    def get_child(
        self,
        next: PseudoTreeNext,
        factory: Dict[Node, "TreeNode"] = None,
        auto_initialize: bool = True,
        on_tree: bool = False,
    ) -> Optional[Tuple["TreeNode", AverageMeter]]:
        """
        Get the child node of a node with a Next. When the node is dead, return None.
        Dependencies: is_dead_end -> is_final -> None
        """
        if on_tree:
            auto_initialize = False
        if self._is_mid:
            assert isinstance(next, int)
            child = self._node.get_child(Next(self._type, next))
            if child is None:
                return None
            if on_tree and (child not in factory or not factory[child]._isin_tree):
                return None
            if child in factory and factory[child].is_dead_end(factory):
                return None
            if child not in factory:
                if auto_initialize:
                    factory[child] = TreeNode(child)
                else:
                    return None
            return factory[child], self.edge_states[next]
        else:
            assert isinstance(next, Next.Type)
            for child in self.children:
                if child._type == next:
                    return (
                        child,
                        child.state
                        if child._isin_tree and not child.is_dead_end(factory)
                        else None,
                    )
            return None

    def is_fully_expanded(self, factory: Dict[Node, "TreeNode"]) -> bool:
        """
        Check whether all visible children have been expanded.
        Dependencies: is_dead_end -> is_final -> None
        Not init new tree nodes.
        """
        if self._is_mid:
            nexts = [
                handle.key
                for handle in self._node.get_children_handles()
                if handle.type == self._type
            ]
            for next in nexts:
                child = self._node.get_child(Next(self._type, next))
                if child is None:
                    continue
                if (
                    child in factory
                    and factory[child]._isin_tree
                    and factory[child].N == 0
                ):
                    return False
            return True
        else:
            return all(
                [child.N > 0 or child.is_dead_end(factory) for child in self.children]
            )

    def is_fully_in_tree(self, factory: Dict[Node, "TreeNode"]) -> bool:
        """
        Check whether all children have been revealed.
        Dependencies: is_dead_end -> is_final -> None
        Not init new tree nodes.
        """
        if self._is_mid:
            nexts = [
                handle.key
                for handle in self._node.get_children_handles()
                if handle.type == self._type
            ]
            for next in nexts:
                child = self._node.get_child(Next(self._type, next))
                if child is None:
                    continue
                if child not in factory or not (
                    factory[child]._isin_tree or factory[child].is_dead_end(factory)
                ):
                    return False
            return True
        else:
            return all(
                [
                    child._isin_tree or child.is_dead_end(factory)
                    for child in self.children
                ]
            )

    def is_terminal(self, factory: Dict[Node, "TreeNode"]) -> bool:
        """
        Check if a node is a terminal (final / dead end).
        Dependencies: is_dead_end -> is_final -> None
        """
        return self.is_final() or self.is_dead_end(factory)

    def set_dead(self) -> None:
        self._is_dead = True

    def is_dead_end(self, factory: Dict[Node, "TreeNode"]) -> bool:
        """
        Check if a node is dead end (will recursively check all expanded children).
        Dependencies: is_final -> None
        """
        if self._is_dead:
            return True
        if self.is_final() and not self.filtered:
            return False
        if self._node.is_dead_end() or self.is_final():
            self._is_dead = True
        if self._is_dead:
            return True

        if self._is_mid:
            for handle in self._node.get_children_handles():
                if handle.type == self._type:
                    child = self._node.get_child(Next(self._type, handle.key))
                    if child and (child not in factory or not factory[child]._is_dead):
                        return False
            self._is_dead = True
        else:
            if len(self.children) == 0 or all(
                [child._is_dead for child in self.children]
            ):
                self._is_dead = True

        return self._is_dead

    def is_exhausted(self, factory: Dict[Node, "TreeNode"]) -> bool:
        """
        Check if a subtree start from a node is exhausted.
        Dependencies: is_dead_end() -> is_final() -> None
        """
        if self._exhausted:
            return True
        if self.is_final() and (self.reward > 0 or self.filtered):
            self._exhausted = True
        if not self.is_fully_in_tree(factory):
            return False
        if self.is_dead_end(factory):
            logging.debug(
                f"exhaust, dead end (internal node state: {self._node.is_dead_end()})"
            )
            self._exhausted = True
        if self._exhausted:
            return True

        if self._is_mid:
            nexts = [
                handle.key
                for handle in self._node.get_children_handles()
                if handle.type == self._type
            ]
            for next in nexts:
                child = self._node.get_child(Next(self._type, next))
                if child is None:
                    continue
                if child not in factory or not factory[child].is_exhausted(factory):
                    return False
            # logging.debug("exhaust, recursive")
            self._exhausted = True
        else:
            non_exhausted_children = [
                child for child in self.children if not child.is_exhausted(factory)
            ]
            if len(non_exhausted_children) == 0:
                self._exhausted = True

        return self._exhausted

    def is_final(self) -> bool:
        """
        Check if a node is final, which means it can be realized as a Halide kernel.
        Dependencies: None
        """
        if self._is_mid:
            return False
        return self._node.is_final()

    def to_node(self) -> "Node":
        """
        Dependencies: None
        """
        return self._node.to_node()

    def __repr__(self) -> str:
        if self._is_mid:
            return str(self._node) + "->" + str(self._type).split(".")[-1]
        return str(self._node)
