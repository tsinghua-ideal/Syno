import random
import math
import logging
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union, Callable, Any

from .Node import Node
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

    def do_rollout(self, node: Node) -> Node:
        "Make the tree one layer better. (Train for one iteration.)"
        while True:
            trial, success = self._may_fail_rollout(node)
            if success:
                logging.debug(
                    f"Successful rollout: {trial.path}. Evaluation to be done.")
                return trial
            else:
                logging.debug(
                    f"During rollout, dead end {trial.path} encountered. Retrying...")
                self.back_propagate(trial, 0.0)

    def get_results(self, node: Node = None) -> Node:
        "Return the best result searched from the tree."
        if node is None:
            node = self._sampler.root()
        while not node.is_terminal():
            node = self._best_select(node)
        return node

    def _may_fail_rollout(self, node: Node) -> Tuple[Node, bool]:
        "The trial may encounter a dead end. In this case, False is returned."
        leaf = self._select(node)
        if leaf.is_dead_end():
            return leaf, False
        self._expand(leaf)
        return self._simulate(leaf)

    def _select(self, node: Node) -> Node:
        "Find an unexplored descendent of `node`"
        while True:
            if node.is_terminal() or (node not in self._children_nexts):
                # node is either terminal or unexplored
                return node
            nexts = self._children_nexts[node]
            for next in nexts:
                augmented = node.get_child(next)
                if augmented not in self._children_nexts:
                    # we found an unexplored descendent of node
                    return augmented
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node: Node) -> None:
        "Update the `children` dict with the children of `node`"
        if node in self._children_nexts:
            return  # already expanded
        self._children_nexts[node] = node.get_children_handles()

    def _simulate(self, node: Node) -> Tuple[Node, bool]:
        "Returns a random simulation (to completion) of `node`"
        node = self._sampler.random_node_with_prefix(node.path)
        return node, not node.is_dead_end()

    # Move increment of `_N` to `do_rollout` for parallel search. TODO
    def back_propagate(self, node: Node, reward: float) -> None:
        "Send the reward back up to the ancestors of the leaf"
        assert 0.0 <= reward <= 1.0

        def _update_stats(node: Node, reward: float) -> None:
            self._N[node] += 1
            self._Q[node] += reward

        path = node.path
        node = self._sampler.root()
        _update_stats(node, reward)
        for next in path:
            node = node.get_child(next)
            _update_stats(node, reward)

    def _uct_select(self, node: Node) -> Node:
        "Select a child of node, balancing exploration & exploitation"

        children = [node.get_child(next) for next in self._children_nexts[node]]

        # All children of node should already be expanded:
        assert all(child in self._children_nexts for child in children)

        log_N_vertex = math.log(self._N[node])

        def uct(child) -> float:
            "Upper confidence bound for trees"
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
                return math.inf  # avoid unseen moves
            return self._Q[n] / self._N[n]  # average reward

        return max(children, key=score)
