import logging
import math
from random import random
from collections import defaultdict
from functools import partial
from typing import List, Union, Optional, Dict, Tuple, DefaultDict, Generator
from time import time

from KAS import Path, Node, AbsolutePath, Sampler, Next, Arc, NextSerializer

from .avg_meter import AverageMeter

PseudoTreeNext = Union[Next.Type, int]
PseudoArc = Union[Next.Type, Arc]

next_serializer = NextSerializer()


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

    def to_path(self) -> Tuple[Path, Optional[Next.Type]]:
        if len(self.abs_path) >= 1 and self.abs_path[-1].key == 0:
            return (
                Path([self.to_next(n) for n in self.abs_path[:-1]]),
                self.abs_path[-1].type,
            )
        else:
            return Path(self.abs_path), None

    @staticmethod
    def deserialize(serialized: str) -> "TreePath":
        deserialized_list = serialized.split("_")
        return TreePath(
            [Next(Next.Type(int(n[0])), int(n[1:])) for n in deserialized_list]
        )

    @staticmethod
    def decode_next_type(repr_: str):
        pos = next_serializer.deserialize_type(repr_)
        return str(pos)

    @staticmethod
    def decode_str(str_repr: str) -> "TreePath":
        str_repr = [r[:-1].split("(") for r in str_repr[1:-1].split(", ")]
        return TreePath(
            [Next(next_serializer.deserialize_type(r[0]), int(r[1])) for r in str_repr]
        )

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

    @property
    def hierarchy(self) -> Generator["TreePath", None, None]:
        path = TreePath([])
        yield path
        for next in self.abs_path:
            path = path.concat(next.type)
            yield path
            if next.key == 0:
                return
            path = path.concat(next.key)
            yield path

    def path_to_strs(self, sampler: Sampler):
        full_path = self.abs_path
        suffix = ""
        if len(self.abs_path) > 0 and self.abs_path[-1].key == 0:
            suffix = str(self.abs_path[-1].type)
            full_path = self.abs_path[:-1]
        return sampler.path_to_strs(
            full_path
        ), suffix if suffix != "" else sampler.path_to_strs(full_path)

    def __repr__(self) -> str:
        if len(self.abs_path) > 0 and self.abs_path[-1].key == 0:
            return f'[{", ".join(str(next) for next in self.abs_path[:-1])}]->{str(self.abs_path[-1].type)[5:]}'
        else:
            return super().__repr__()


class TreeNode:
    """
    A wrapper of node that represents either a type or a full node
    """

    def __init__(
        self, tree, node: Node, is_mid: bool = False, type: Next.Type = None
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
        self._tree = tree
        self._node: Node = node
        self._is_mid: bool = is_mid
        self._type: Next.Type = type

        # states
        self.state = AverageMeter(support_std=True)
        self._last_T: int = 0
        self._is_dead: bool = False
        self._simulate_fail: bool = False
        self._not_dead: bool = False
        self._exhausted: bool = False
        self._isin_tree: bool = False
        self._virtual_loss: int = 0

        # temporal buffer
        self._simulate_attempt_time: float = 0
        self._failed_budget: float = 5

        self.l_rave: DefaultDict[PseudoArc, AverageMeter] = defaultdict(AverageMeter)

        # conditional members
        if node.is_final():
            self.reward: float = -1.0
            self.filtered: bool = False
        if self._is_mid:
            self.edge_states: DefaultDict[int, AverageMeter] = defaultdict(
                partial(AverageMeter, support_std=True)
            )
        else:
            self.children: List["TreeNode"] = []
            primitives = self._node.collect_operations()
            for child in primitives.keys():
                self.children.append(
                    TreeNode(self._tree, node, is_mid=True, type=child)
                )

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

    def eq_state(self, __value: "TreeNode"):
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

    def __eq__(self, __value: "TreeNode") -> bool:
        return (
            self._node == __value._node
            and self._is_mid == __value._is_mid
            and self._type == __value._type
        )

    def __hash__(self) -> int:
        return hash((self._node, self._is_mid, self._type))

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
            "_is_not_dead": self._not_dead,
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
            and not self._not_dead
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
        self._not_dead = state_dict["_is_not_dead"]
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
        if self._simulate_fail:
            logging.debug(f"Resurrected {self} during lrave update. ")
            self._simulate_fail = False
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
        filter_exhausted: bool = False,
    ) -> None:
        """
        Flush T (for progressive widening).
        Dependencies:
            children_count -> exhausted -> is_dead_end -> is_final
            is_fully_in_tree -> is_dead_end -> is_final
            reveal_new_children -> get_unrevealed_children -> get_children -> get_child -> is_dead_end -> is_final
        """
        Tp = math.floor(T**self._tree._b)
        # print(f"Got Tp={Tp}, child_count={self.children_count(on_tree=True, filter_exhausted=filter_exhausted)}, i {'am' if self.is_fully_in_tree() else 'am not'} fully in tree")
        while Tp > self.children_count(
            on_tree=True,
            filter_exhausted=filter_exhausted,
            filter_simulate_failure=True,
        ):
            if self.is_fully_in_tree():
                break
            else:
                self.reveal_new_children()
        self._last_T = T

    def children_count(
        self,
        include_uninitialize: bool = False,
        on_tree: bool = False,
        filter_exhausted: bool = False,
        filter_simulate_failure: bool = False,
    ) -> int:
        """
        Get the number of all children of a node.
        Dependencies: is_exhausted -> is_dead_end -> is_final -> None
        """
        assert not (include_uninitialize and on_tree)
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
                if include_uninitialize and child not in self._tree._treenode_store:
                    count += 1
                    continue
                if (
                    on_tree
                    and child in self._tree._treenode_store
                    and not self._tree._treenode_store[child]._isin_tree
                ):
                    continue
                if (
                    filter_simulate_failure
                    and child in self._tree._treenode_store
                    and self._tree._treenode_store[child]._simulate_fail
                ):
                    continue
                if filter_exhausted:
                    if (
                        child in self._tree._treenode_store
                        and not self._tree._treenode_store[child].is_exhausted()
                    ):
                        count += 1
                else:
                    if (
                        child in self._tree._treenode_store
                        and not self._tree._treenode_store[child].is_dead_end()
                    ):
                        count += 1
            return count
        else:
            if filter_exhausted:
                return len(self.children) - sum(
                    (
                        child.is_exhausted()
                        or (on_tree and not child._isin_tree)
                        or (filter_simulate_failure and child._simulate_fail)
                    )
                    for child in self.children
                )
            else:
                return len(self.children) - sum(
                    (
                        child.is_dead_end()
                        or (on_tree and not child._isin_tree)
                        or (filter_simulate_failure and child._simulate_fail)
                    )
                    for child in self.children
                )

    def reveal_new_children(self) -> bool:
        """
        Reveal a new children to its parent.
        Return success or not.

        Dependencies: get_unrevealed_children -> get_children -> get_child -> is_dead_end -> is_final -> None
        """

        def rave_encoder(rave: AverageMeter):
            if rave.N == 0:
                return -1.0
            return rave.mean + rave.std - 1 / rave.N

        grave_list = []
        lrave_list = []
        beta_list = []
        children = self.get_children(auto_initialize=True)
        for next, _, _ in children:
            if self._is_mid:
                assert isinstance(next, int)
                arc = self._node.get_arc_from_handle(Next(self._type, next))
                if arc is None:  # the next is already dead
                    grave_list.append(-100.0)
                    lrave_list.append(-100.0)
                    beta_list.append(-100.0)
                    continue
            else:
                assert isinstance(next, Next.Type)
                arc = next

            grave = (
                rave_encoder(self._tree.g_rave[arc])
                if arc in self._tree.g_rave
                else -1.0
            )
            lrave = rave_encoder(self.l_rave[arc]) if arc in self.l_rave else -1.0
            beta = (
                self._tree._c_l / (self._tree._c_l + self.l_rave[arc].N)
                if arc in self.l_rave
                else 1.0
            )
            grave_list.append(grave)
            lrave_list.append(lrave)
            beta_list.append(beta)

        def replace_mean(
            lst: List[float], placeholder: float, ignored_value: float
        ) -> List[float]:
            counted_elems = [x for x in lst if x not in [placeholder, ignored_value]]
            mean_value = (
                sum(counted_elems) / len(counted_elems)
                if len(counted_elems) > 0
                else 0.0
            )
            return [mean_value if x == placeholder else x for x in lst]

        grave_list = replace_mean(grave_list, -1.0, -100.0)
        lrave_list = replace_mean(lrave_list, -1.0, -100.0)

        rave_scores = [
            ((1 - beta) * l_rave + beta * g_rave) * (1 - self._tree._rave_random_ratio)
            + self._tree._rave_random_ratio * random()
            for g_rave, l_rave, beta in zip(grave_list, lrave_list, beta_list)
        ]
        unrevealed_children_with_rave = [
            (child, rave_score)
            for (_, child, _), rave_score in zip(children, rave_scores)
            if not child._isin_tree
        ]
        if len(unrevealed_children_with_rave) == 0:
            logging.debug("No new children to be added")
            assert (
                self.is_fully_in_tree()
            ), f"{self} is not fully expanded but no children can be revealed"
            return False
        selected_child, score = max(unrevealed_children_with_rave, key=lambda x: x[1])
        if score == -100.0:
            return False
        selected_child._isin_tree = True
        return True

    def get_unrevealed_children(
        self,
    ) -> List[Tuple[PseudoTreeNext, "TreeNode", AverageMeter]]:
        """
        Get all unrevealed children of a node with nexts and edge states.
        Dependencies: get_children -> get_child -> is_dead_end -> is_final -> None
        """
        children = self.get_children(auto_initialize=True)
        unrevealed_children = [child for child in children if not child[1]._isin_tree]
        return unrevealed_children

    def get_unexpanded_children(
        self,
    ) -> List[Tuple[PseudoTreeNext, "TreeNode", AverageMeter]]:
        """
        Get all unexpanded children of a node with nexts and edge states.
        Dependencies: get_children -> get_child -> is_dead_end -> is_final -> None
        """
        children = self.get_children(on_tree=True, filter_simulate_failure=True)
        unexpanded_children = [child for child in children if child[1].N == 0]
        return unexpanded_children

    def get_children_nexts(self) -> List[PseudoTreeNext]:
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
                and not (
                    (ch in self._tree._treenode_store)
                    and (self._tree._treenode_store[ch].is_dead_end())
                )
            ]
        else:
            nexts = [child._type for child in self.children if not child.is_dead_end()]

        return nexts

    def get_children(
        self,
        auto_initialize: bool = False,
        on_tree: bool = False,
        filter_simulate_failure: bool = True,
    ) -> List[Tuple[PseudoTreeNext, "TreeNode", AverageMeter]]:
        """
        Get all alive children of a node with nexts and edge states.
        Dependencies: get_child -> is_dead_end -> is_final -> None
        """
        assert not (on_tree and auto_initialize)

        if self._is_mid:
            nexts = [
                handle.key
                for handle in self._node.get_children_handles()
                if handle.type == self._type
            ]
            children = [
                self.get_child(
                    nxt,
                    auto_initialize=auto_initialize,
                    on_tree=on_tree,
                    filter_simulate_failure=filter_simulate_failure,
                )
                for nxt in nexts
            ]

            # Remove filtered and dead children.
            filtered = [
                (nxt, c[0], c[1]) for c, nxt in zip(children, nexts) if c is not None
            ]
            nexts = [nxt for nxt, _, _ in filtered]
            children = [c for _, c, _ in filtered]
            edge_states = [edge for _, _, edge in filtered]
        else:
            children = self.children
            children = [child for child in children if not child.is_dead_end()]
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
        auto_initialize: bool = False,
        on_tree: bool = False,
        filter_simulate_failure: bool = True,
    ) -> Optional[Tuple["TreeNode", AverageMeter]]:
        """
        Get the child node of a node with a Next. When the node is dead, return None.
        Dependencies: is_dead_end -> is_final -> None
        """
        assert not (on_tree and auto_initialize)
        if self._is_mid:
            assert isinstance(next, int)
            child = self._node.get_child(Next(self._type, next))
            if child is None:
                return None
            if on_tree and (
                child not in self._tree._treenode_store
                or not self._tree._treenode_store[child]._isin_tree
            ):
                return None
            if child in self._tree._treenode_store and self._tree._treenode_store[
                child
            ].is_dead_end(filter_simulate_failure=filter_simulate_failure):
                return None
            if child not in self._tree._treenode_store:
                if auto_initialize:
                    self._tree.touch(
                        child,
                        path=self._tree._path_store[self._node].concat(
                            Next(self._type, next)
                        ),
                    )
                else:
                    return None
            return self._tree._treenode_store[child], self.edge_states[next]
        else:
            assert isinstance(next, Next.Type)
            for child in self.children:
                if child._type == next:
                    return (
                        child,
                        child.state
                        if (not on_tree or child._isin_tree)
                        and not child.is_dead_end(
                            filter_simulate_failure=filter_simulate_failure
                        )
                        else None,
                    )
            return None

    def is_fully_expanded(self) -> bool:
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
                    child in self._tree._treenode_store
                    and self._tree._treenode_store[child]._isin_tree
                    and self._tree._treenode_store[child].N == 0
                ):
                    return False
            return True
        else:
            return all([child.N > 0 or child.is_dead_end() for child in self.children])

    def is_fully_in_tree(self) -> bool:
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
                if child not in self._tree._treenode_store or not (
                    self._tree._treenode_store[child]._isin_tree
                    or self._tree._treenode_store[child].is_dead_end()
                ):
                    return False
            return True
        else:
            return all(
                [child._isin_tree or child.is_dead_end() for child in self.children]
            )

    def is_terminal(self) -> bool:
        """
        Check if a node is a terminal (final / dead end).
        Dependencies: is_dead_end -> is_final -> None
        """
        return self.is_final() or self.is_dead_end()

    def set_dead(self) -> None:
        self._is_dead = True

    def set_simulate_fail(self) -> None:
        self._simulate_fail = True

    def set_alive(self) -> None:
        self._not_dead = True

    def is_alive(self) -> None:
        return self._not_dead

    def is_dead_end(self, filter_simulate_failure=True) -> bool:
        """
        Check if a node is dead end (will recursively check all expanded children).
        Dependencies: is_final -> None
        """
        if self._is_dead:
            return True
        if filter_simulate_failure and self._simulate_fail:
            return True
        if self.is_final():
            if self.filtered or self._node.is_dead_end():
                self._is_dead = True
                return True
            else:
                return False

        if self._is_mid:
            for handle in self._node.get_children_handles():
                if handle.type == self._type:
                    child = self._node.get_child(Next(self._type, handle.key))
                    # print("child", child, self._tree._treenode_store, (child in self._tree._treenode_store))
                    if child and not (
                        child in self._tree._treenode_store
                        and self._tree._treenode_store[child]._is_dead
                    ):
                        return False
            self._is_dead = True
        else:
            if len(self.children) == 0 or all(
                [child._is_dead for child in self.children]
            ):
                self._is_dead = True

        return self._is_dead

    def is_exhausted(self) -> bool:
        """
        Check if a subtree start from a node is exhausted.
        Dependencies: is_dead_end() -> is_final() -> None
        """
        if self._exhausted:
            return True
        if self.is_final() and (self.reward > 0 or self.filtered):
            self._exhausted = True
        if not self.is_fully_in_tree():
            return False
        if self.is_dead_end():
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
                if (
                    child not in self._tree._treenode_store
                    or not self._tree._treenode_store[child].is_exhausted()
                ):
                    return False
            # logging.debug("exhaust, recursive")
            self._exhausted = True
        else:
            non_exhausted_children = [
                child for child in self.children if not child.is_exhausted()
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
        return self._node

    def __repr__(self) -> str:
        if self._is_mid:
            return str(self._node) + "->" + str(self._type).split(".")[-1]
        return str(self._node)
