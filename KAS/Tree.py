import logging
from collections import defaultdict
import math
import random

from .Sampler import Sampler


class MCTS:
    def __init__(self, sampler: Sampler, exploration_weight: float = math.sqrt(2)):
        self._Q = defaultdict(float)  # Total reward of each node.
        self._N = defaultdict(int)  # Total visit count for each node.
        self._children_count = dict()  # Number of children of each node. Only explored nodes are in this dict.
        self._sampler = sampler
        self._exploration_weight = exploration_weight
        random.seed(sampler._seed)

    def do_rollout(self, node: list[int]) -> list[int]:
        "Make the tree one layer better. (Train for one iteration.)"
        while True:
            path, success = self._may_fail_rollout(node)
            if success:
                logging.debug(f"Successful rollout: {path}. Evaluation to be done.")
                return path
            else:
                logging.warning(f"During rollout, dead end {path} encountered. Retrying...")
                self.back_propagate(path, 0.0)

    def _may_fail_rollout(self, node: list[int]) -> (list[int], bool):
        "The trial may encounter a dead end. In this case, False is returned."
        leaf = self._select(node)
        if self._sampler.is_dead_end(leaf):
            return leaf, False
        self._expand(leaf)
        return self._random_path(leaf)

    def _is_terminal(self, node: list[int]) -> bool:
        return self._sampler.is_terminal(node)
    
    def _is_dead_end(self, node: list[int]) -> bool:
        return self._sampler.is_dead_end(node)

    def _compute_children_count(self, node: list[int]) -> int:
        return self._sampler.children_count(node)

    def _select(self, node: list[int]) -> list[int]:
        "Find an unexplored descendent of `node`"
        while True:
            tuple_node = tuple(node)
            if self._is_terminal(node) or (tuple_node not in self._children_count):
                # node is either terminal or unexplored
                return node
            cnt = self._children_count[tuple_node]
            for last in range(cnt):
                augmented = node + [last]
                if tuple(augmented) not in self._children_count:
                    # we found an unexplored descendent of node
                    return augmented
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node: list[int]):
        "Update the `children` dict with the children of `node`"
        tuple_node = tuple(node)
        if tuple_node in self._children_count:
            return # already expanded
        # Terminal nodes have no children. Actually an arbitrary value is fine for terminals.
        self._children_count[tuple_node] = 0 if self._is_terminal(node) else self._compute_children_count(node)

    def _random_path(self, node: list[int]) -> (list[int], bool):
        "Returns a random simulation (to completion) of `node`"
        while True:
            if self._is_terminal(node):
                return node, not self._is_dead_end(node)
            node = node + [random.randrange(0, self._compute_children_count(node))]

    # Move increment of `_N` to `do_rollout` for parallel search. TODO
    def back_propagate(self, node: list[int], reward: float):
        "Send the reward back up to the ancestors of the leaf"
        assert 0.0 <= reward <= 1.0
        for i in range(len(node), -1, -1):
            tuple_node = tuple(node[:i])
            self._N[tuple_node] += 1
            self._Q[tuple_node] += reward

    def _uct_select(self, node: list[int]) -> list[int]:
        "Select a child of node, balancing exploration & exploitation"

        tuple_node = tuple(node)
        # All children of node should already be expanded:
        assert all(tuple(node + [n]) in self._children_count for n in range(self._children_count[tuple_node]))

        log_N_vertex = math.log(self._N[tuple_node])

        def uct(n):
            "Upper confidence bound for trees"
            t = tuple(node + [n])
            return self._Q[t] / self._N[t] + self._exploration_weight * math.sqrt(
                log_N_vertex / self._N[t]
            )

        return node + [max(self._children_count[tuple_node], key=uct)]
