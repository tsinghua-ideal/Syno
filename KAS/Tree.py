import random
import math
import logging
from collections import defaultdict
from typing import List, Tuple, Any

from .Node import Path, VisitedNode
from .Sampler import Sampler
from .TreeNode import TreeNode, TreePath
from .Bindings import Next


class MCTS:
    def __init__(self, sampler: Sampler, virtual_loss_constant: float = 0.0, leaf_num: int = 1, simulate_retry_limit: int = 10, exploration_weight: float = math.sqrt(2)) -> None:
        self._Q = defaultdict(float)  # Total reward of each node.
        self._N = defaultdict(int)  # Total visit count for each node.
        # Number of children of each node. Only explored nodes are in this dict.
        self._children_nexts = dict()
        self._sampler = sampler
        self._exploration_weight = exploration_weight
        random.seed(sampler._seed)

        self.virtual_loss_constant = virtual_loss_constant
        self.virtual_loss_count = defaultdict(
            lambda: 0)  # node -> virtual loss count
        self.leaf_num = leaf_num
        self.simulate_retry_limit = simulate_retry_limit

    def dump(self) -> dict:
        """Dump the tree into a json file. """
        Q_sel = {k.path.serialize(): v for k, v in self._Q.items()}
        N_sel = {k.path.serialize(): v for k, v in self._N.items()}
        node_expanded = list(self._children_nexts.keys())
        node_expanded_sel = [n.path.serialize() for n in node_expanded]
        packed_args = dict(
            virtual_loss_constant=self.virtual_loss_constant,
            leaf_num=self.leaf_num,
            simulate_retry_limit=self.simulate_retry_limit,
            _exploration_weight=self._exploration_weight
        )
        j = dict(
            Q=Q_sel,
            N=N_sel,
            node_expand=node_expanded_sel,
            args=packed_args
        )
        return j

    def _get_Q(self, node: TreeNode) -> float:
        return self._Q[node]

    def _add_Q(self, node: TreeNode, value: float) -> None:
        self._Q[node] += value

    def _get_N(self, node: TreeNode) -> int:
        return self._N[node]

    def _add_N(self, node: TreeNode, value: int) -> None:
        self._N[node] += value

    def _has_children_nexts(self, node: TreeNode) -> bool:
        return node in self._children_nexts

    def _get_children_nexts(self, node: TreeNode) -> List[Any]:
        return self._children_nexts[node]

    def _set_children_nexts(self, node: TreeNode, children_nexts: List[Any]) -> None:
        self._children_nexts[node] = children_nexts

    def _increment_virtual_loss(self, path: TreePath, node: TreeNode):
        node = TreeNode(node.path, node._node)
        for next in path:
            node = node.get_child(next.type)
            self.virtual_loss_count[node] += 1
            if next.key == 0:
                break
            node = node.get_child(next.key)
            self.virtual_loss_count[node] += 1

    def _decrement_virtual_loss(self, path: TreePath, node: TreeNode):
        node = TreeNode(node.path, node._node)
        for next in path:
            node = node.get_child(next.type)
            self.virtual_loss_count[node] -= 1
            assert self.virtual_loss_count[node] >= 0
            if next.key == 0:
                break
            node = node.get_child(next.key)
            self.virtual_loss_count[node] -= 1

        assert self.virtual_loss_count[node] >= 0

    # The receipt which is used for back propagation, which is the root node and the path.
    Receipt = Tuple[TreeNode, TreePath]

    def do_rollout(self, node: VisitedNode) -> Tuple[Receipt, List[VisitedNode]]:
        "Make the tree one layer better. (Train for one iteration.)"
        "Returns `((root, path), leaf)`. `path` is used to propagate reward. `leaf` is the result of simulation. Note that following `path` we do not necessarily arrive at `leaf`."
        while True:
            path, trials, success = self._may_fail_rollout(node)
            if success:
                logging.debug(
                    f"Successful rollout: {path}. Evaluation to be done.")
                self._increment_virtual_loss(path, node)
                return (node, path), trials
            else:
                logging.debug(
                    f"During rollout, dead end {path} encountered. Retrying...")
                # This kind of interferes with the evaluation. TODO
                self.back_propagate((node, path), 0.0)

    def get_results(self, node: VisitedNode = None) -> VisitedNode:
        "Return the best result searched from the tree."
        if node is None:
            node = self._sampler.root()
        node = TreeNode(node.path, node._node, is_mid=True, type=node.path[-1].type) if len(
            node.path) > 0 and node.path[-1] == 0 else TreeNode(node.path, node._node)
        while not node.is_terminal():
            node = self._best_select(node)
        return node

    def _may_fail_rollout(self, node: VisitedNode) -> Tuple[TreePath, List[TreeNode], bool]:
        "The trial may encounter a dead end. In this case, False is returned."
        node = TreeNode(node.path, node._node, is_mid=True, type=node.path[-1].type) if len(
            node.path) > 0 and node.path[-1] == 0 else TreeNode(node.path, node._node)
        path, leaf = self._select(node)
        if leaf.is_dead_end():
            return path, leaf, False
        self._expand(leaf)
        leaves = []
        for _ in range(self.simulate_retry_limit):
            leaf_simul, success = self._simulate(leaf)
            if success:
                leaves.append(leaf_simul)
                if len(leaves) == self.leaf_num:
                    break
        success = len(leaves) == self.leaf_num
        return path, leaves, success

    def _select(self, node: TreeNode) -> Tuple[TreePath, TreeNode]:
        "Find an unexplored descendent of `node`"
        # Here, the path is just arbitrary, and depends on how we build the search tree. See doc for `_uct_select`. We only need to make sure we can construct a `TwoStepPath` from `path`.
        path = TreePath([])
        while True:
            if node.is_terminal() or (not self._has_children_nexts(node)):
                # node is either terminal or unexplored
                return path, node
            nexts = self._get_children_nexts(node)
            for next in nexts:
                augmented = node.get_child(next)
                if not self._has_children_nexts(augmented):
                    # we found an unexplored descendent of node
                    path = path.concat(next)
                    logging.debug(f'path is {path}')
                    return TreePath(path), augmented
            next, node = self._uct_select(node)  # descend a layer deeper
            path = path.concat(next)

    def _expand(self, node: TreeNode) -> None:
        "Update the `children` dict with the children of `node`"
        if self._has_children_nexts(node):
            return  # already expanded
        self._set_children_nexts(node, node.get_children_handles())

    def _simulate(self, node: TreeNode) -> Tuple[TreeNode, bool]:
        "Returns a random simulation (to completion) of `node`"

        def random_child(node: TreeNode) -> TreeNode:
            """
            Two step random selection. First, randomly select a primitive type. Then, randomly select a child of that type.
            """
            selected_child = random.choice(node.get_children_handles())
            return node.get_child(selected_child)

        while not node.is_terminal():
            node = random_child(node)
        return node, node.is_final()

    # Move increment of `_N` to `do_rollout` for parallel search. TODO
    def back_propagate(self, receipt: Receipt, reward: float) -> None:
        "Send the reward back up to the ancestors of the leaf"
        assert 0.0 <= reward <= 1.0

        def _update_stats(node: TreeNode, reward: float) -> None:
            self._add_N(node, 1)
            self._add_Q(node, reward)

        node, path = receipt
        node = TreeNode(node.path, node._node)
        if reward > 0:
            self._decrement_virtual_loss(path, node)

        _update_stats(node, reward)
        for next in path:
            node = node.get_child(next.type)
            _update_stats(node, reward)
            if next.key == 0:
                break
            node = node.get_child(next.key)
            _update_stats(node, reward)

    # Here, the returned Any is just an element in the path. The type depends on how we build the search tree. If we follow the C++ implementation, it should be a kas_cpp_bindings.Next. If we use two-step generation for primitives, it should be either Union[primitive type, index of primitive], which is not yet implemented.
    def _uct_select(self, node: TreeNode) -> Tuple[Any, TreeNode]:
        "Select a child of node, balancing exploration & exploitation"

        nexts = self._get_children_nexts(node)
        children = [(next, node.get_child(next)) for next in nexts]

        # All children of node should already be expanded:
        assert all(self._has_children_nexts(child) for _, child in children)

        if self._get_N(node) > 0:
            log_N_vertex = math.log(self._get_N(node))

        def uct(child) -> float:
            "Upper confidence bound for trees"
            _, child = child
            if self._get_N(child) == 0:
                return -math.inf  # avoid unseen moves
            return self._get_Q(child) / self._get_N(child) + self._exploration_weight * math.sqrt(
                log_N_vertex / self._get_N(child)
            ) - self.virtual_loss_constant * self.virtual_loss_count[child]

        return max(children, key=uct)

    def _best_select(self, node: TreeNode) -> TreeNode:
        "Select the best child of a given node"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if not self._has_children_nexts(node):
            children_nexts = node.get_children_handles()
            return node.get_child(random.choice(children_nexts))

        children = [node.get_child(next)
                    for next in self._get_children_nexts(node)]

        def score(n) -> float:
            if self._get_N(n) == 0:
                return -1  # avoid unseen moves
            return self._get_Q(n) / self._get_N(n)  # average reward

        return max(children, key=score)
