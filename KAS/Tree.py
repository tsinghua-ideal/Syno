import random
import math
import logging
from collections import defaultdict
from typing import List, Tuple, Any

from .Node import Path, Node
from .Sampler import Sampler


class MCTS:
    def __init__(self, sampler: Sampler, exploration_weight: float = math.sqrt(2)) -> None:
        self._Q = defaultdict(float)  # Total reward of each node.
        self._N = defaultdict(int)  # Total visit count for each node.
        # Number of children of each node. Only explored nodes are in this dict.
        self._children_nexts = dict()
        self._sampler = sampler
        self._exploration_weight = exploration_weight
        random.seed(sampler._seed)

    # The receipt which is used for back propagation, which is the root node and the path.
    Receipt = Tuple[Node, Path]

    def do_rollout(self, node: Node) -> Tuple[Receipt, Node]:
        "Make the tree one layer better. (Train for one iteration.)"
        "Returns `((root, path), leaf)`. `path` is used to propagate reward. `leaf` is the result of simulation. Note that following `path` we do not necessarily arrive at `leaf`."
        while True:
            path, trial, success = self._may_fail_rollout(node)
            if success:
                logging.debug(
                    f"Successful rollout: {path}. Evaluation to be done.")
                return (node, path), trial
            else:
                logging.debug(
                    f"During rollout, dead end {path} encountered. Retrying...")
                # This kind of interferes with the evaluation. TODO
                self.back_propagate((node, path), 0.0)

    def get_results(self, node: Node = None) -> Node:
        "Return the best result searched from the tree."
        if node is None:
            node = self._sampler.root()
        while not node.is_terminal():
            node = self._best_select(node)
        return node

    def _may_fail_rollout(self, node: Node) -> Tuple[Path, Node, bool]:
        "The trial may encounter a dead end. In this case, False is returned."
        path, leaf = self._select(node)
        if leaf.is_dead_end():
            return path, leaf, False
        self._expand(leaf)
        leaf, success = self._simulate(leaf)
        return path, leaf, success

    def _select(self, node: Node) -> Tuple[Path, Node]:
        "Find an unexplored descendent of `node`"
        # Here, the path is just arbitrary, and depends on how we build the search tree. See doc for `_uct_select`. We only need to make sure we can construct a `Path` from `path`.
        path: List[Any] = []
        while True:
            if node.is_terminal() or (node not in self._children_nexts):
                # node is either terminal or unexplored
                return path, node
            nexts = self._children_nexts[node]
            for next in nexts:
                augmented = node.get_child(next)
                if augmented not in self._children_nexts:
                    # we found an unexplored descendent of node
                    path.append(next)
                    return Path(path), augmented
            next, node = self._uct_select(node)  # descend a layer deeper
            path.append(next)

    def _expand(self, node: Node) -> None:
        "Update the `children` dict with the children of `node`"
        if node in self._children_nexts:
            return  # already expanded
        self._children_nexts[node] = node.get_children_handles()

    def _simulate(self, node: Node) -> Tuple[Node, bool]:
        "Returns a random simulation (to completion) of `node`"
        def random_child(node: Node) -> Node:
            next = random.choice(node.get_children_handles())
            return node.get_child(next)
        while not node.is_terminal():
            node = random_child(node)
        return node, node.is_final()

    # Move increment of `_N` to `do_rollout` for parallel search. TODO
    def back_propagate(self, receipt: Receipt, reward: float) -> None:
        "Send the reward back up to the ancestors of the leaf"
        assert 0.0 <= reward <= 1.0

        def _update_stats(node: Node, reward: float) -> None:
            self._N[node] += 1
            self._Q[node] += reward

        node, path = receipt
        _update_stats(node, reward)
        for next in path:
            node = node.get_child(next)
            _update_stats(node, reward)

    # Here, the returnded Any is just an element in the path. The type depends on how we build the search tree. If we follow the C++ implementation, it should be a kas_cpp_bindings.Next. If we use two-step generation for primitives, it should be either Union[primitive type, index of primitive], which is not yet implemented.
    def _uct_select(self, node: Node) -> Tuple[Any, Node]:
        "Select a child of node, balancing exploration & exploitation"

        nexts = self._children_nexts[node]
        children = [(next, node.get_child(next)) for next in nexts]

        # All children of node should already be expanded:
        assert all(child in self._children_nexts for _, child in children)

        log_N_vertex = math.log(self._N[node])

        def uct(child) -> float:
            "Upper confidence bound for trees"
            _, child = child
            if self._N[child] == 0:
                return -1  # avoid unseen moves
            return self._Q[child] / self._N[child] + self._exploration_weight * math.sqrt(
                log_N_vertex / self._N[child]
            )

        return max(children, key=uct)

    def _best_select(self, node: Node) -> Node:
        "Select the best child of a given node"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self._children_nexts:
            children_nexts = node.get_children_handles()
            return node.get_child(random.choice(children_nexts))

        children = [node.get_child(next) for next in self._children_nexts[node]]

        def score(n) -> float:
            if self._N[n] == 0:
                return -1  # avoid unseen moves
            return self._Q[n] / self._N[n]  # average reward

        return max(children, key=score)
