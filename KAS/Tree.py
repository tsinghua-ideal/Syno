import random
import math
import logging
from collections import defaultdict, OrderedDict
from typing import List, Tuple, Any, Optional, Dict

from .Node import Path, VisitedNode, Node
from .Sampler import Sampler
from .TreeNode import TreeNode, TreePath, PseudoTreeNext
from .Bindings import Next
from .Utils import NextSerializer


class MCTS:
    def __init__(self, sampler: Sampler, virtual_loss_constant: float = 0.0, leaf_num: int = 1, exploration_weight: float = math.sqrt(2)) -> None:

        # Dict[Node, TreeNode]
        self._treenode_store: Dict[Node, TreeNode] = OrderedDict()
        root = sampler.visit([])
        self._treenode_store[root.to_node()] = TreeNode(root.to_node())

        self._sampler = sampler
        self._exploration_weight = exploration_weight
        random.seed(sampler._seed)

        # Tree Parallelization
        self.virtual_loss_constant = virtual_loss_constant
        self.virtual_loss_count: Dict[TreeNode, int] = defaultdict(
            int)  # node -> virtual loss count

        # Leaf Parallelization
        self.leaf_num = leaf_num
        
        self.next_serializer = NextSerializer()

    def _increment_virtual_loss(self, path: TreePath, node: Node, delta: int=1) -> None:
        assert delta > 0
        tree_node = self._treenode_store[node.to_node()]
        for next in path:
            tree_node = tree_node.get_child(next.type, self._treenode_store)
            assert tree_node is not None
            self.virtual_loss_count[tree_node] += delta
            if next.key == 0:
                break
            tree_node = tree_node.get_child(next.key, self._treenode_store)
            assert tree_node is not None
            self.virtual_loss_count[tree_node] += delta

    def _decrement_virtual_loss(self, path: TreePath, node: Node, delta: int=1) -> None:
        assert delta > 0
        tree_node = self._treenode_store[node.to_node()]
        for next in path:
            tree_node = tree_node.get_child(next.type, self._treenode_store)
            assert tree_node is not None
            self.virtual_loss_count[tree_node] -= 1
            assert self.virtual_loss_count[tree_node] >= 0
            if next.key == 0:
                break
            tree_node = tree_node.get_child(next.key, self._treenode_store)
            assert tree_node is not None
            self.virtual_loss_count[tree_node] -= 1
            assert self.virtual_loss_count[tree_node] >= 0

    # The receipt which is used for back propagation, which is the root node and the path.
    Receipt = Tuple[Node, TreePath]

    def do_rollout(self, node: VisitedNode) -> Optional[Tuple[Receipt, List[TreeNode]]]:
        """
        Make the tree one layer better. (Train for one iteration.)

        Returns `((root, path), leaf)`. `path` is used to propagate reward. `leaf` is the result of simulation. Note that following `path` we do not necessarily arrive at `leaf`.
        
        If the tree is exhausted, then return None. 
        """
        logging.debug(
            f"Rolling out from {node}.")
        while True:
            if self._treenode_store[node.to_node()].is_dead_end(self._treenode_store):
                logging.info("The tree is exhausted. ")
                return None
            result, success = self._may_fail_rollout(node)
            if success:
                path, trials = result
                logging.debug(
                    f"Successful rollout: {path}. Evaluation to be done.")
                self._increment_virtual_loss(path, node, len(trials))
                return (node.to_node(), path), trials
            else:
                logging.debug(
                    f"During rollout, dead end encountered. Retrying...")

    def _may_fail_rollout(self, node: VisitedNode) -> Tuple[Optional[Tuple[TreePath, List[TreeNode]]], bool]:
        "The trial may encounter a dead end. In this case, False is returned."
        tree_node = self._treenode_store[node.to_node()]

        # Select
        logging.debug("Selection start")
        select_result = self._select(tree_node)
        if select_result is None:
            return None, False
        path, leaf = select_result
        logging.debug(f"Selection end {path} {leaf}")

        # Expand
        # TODO: what if the selected node is terminal? 
        logging.debug("Expansion start")
        expand_result = self._expand(leaf)
        if expand_result is None:
            return None, False
        next_expand, leaf_expanded = expand_result
        path = path.concat(next_expand)
        assert isinstance(path, TreePath), type(path)
        logging.debug(f"Expansion end {path} {leaf_expanded}")

        # Simulate
        leaves = []
        logging.debug(f"Simulation start")
        while not leaf_expanded.is_dead_end(self._treenode_store):
            leaf_simul, success = self._simulate(leaf_expanded)
            if success:
                leaves.append(leaf_simul)
                if len(leaves) == self.leaf_num:
                    break
        if leaf_expanded.is_dead_end(self._treenode_store):
            return None, False

        return (path, leaves), success

    def _select(self, node: TreeNode) -> Optional[Tuple[TreePath, TreeNode]]:
        "Find an unexplored descendent of `node`"
        # Here, the path is just arbitrary, and depends on how we build the search tree. See doc for `_uct_select`. We only need to make sure we can construct a `TwoStepPath` from `path`.
        path = TreePath([])
        while True:
            if node.is_terminal() or len(node.get_unexpanded_children(self._treenode_store)) > 0:
                # node is either terminal or unexplored
                return path, node
            selected = self._ucd_select(node)  # descend a layer deeper
            if selected is None:
                return None
            next, node = selected
            path = path.concat(next)

    def _expand(self, node: TreeNode) -> Optional[Tuple[PseudoTreeNext, TreeNode]]:
        """
        Expand the leaf one level deeper, by choosing a random unexpanded child (N=0). Return None if failed. 
        """

        unexpanded_children = node.get_unexpanded_children(self._treenode_store)
        if len(unexpanded_children) == 0:
            logging.debug("Expand failed.")
            return None

        # randomly select a child from pool
        next, leaf = random.choice(unexpanded_children)
        return next, leaf

    def _simulate(self, node: TreeNode) -> Tuple[Optional[TreeNode], bool]:
        "Returns a random simulation (to completion) of `node`"

        def random_child(node: TreeNode) -> Optional[TreeNode]:
            """
            Two step random selection. First, randomly select a primitive type. Then, randomly select a child of that type.
            """
            assert isinstance(node, TreeNode)
            children = node.get_children(self._treenode_store)
            logging.debug(f"{node} has children {children}")
            if len(children) == 0:
                return None
            next, selected_child = random.choice(children)
            logging.debug(f"Random selected {next}: {selected_child}")
            return selected_child

        while not node.is_terminal():
            node = random_child(node)
            if node is None:
                return None, False
        return node, node.is_final()

    def back_propagate(self, receipt: Receipt, reward: float) -> None:
        "Send the reward back up to the ancestors of the leaf"
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0

        def _update_stats(node: TreeNode, reward: float) -> None:
            assert isinstance(node, TreeNode)
            assert isinstance(reward, float)
            node.N += 1
            node.Q += reward

        node, path = receipt
        self._decrement_virtual_loss(path, node)

        node = self._treenode_store[node.to_node()]
        _update_stats(node, reward)
        for next in path:
            node = node.get_child(next.type, self._treenode_store)
            assert node is not None
            _update_stats(node, reward)
            if next.key == 0:
                break
            node = node.get_child(next.key, self._treenode_store)
            assert node is not None
            _update_stats(node, reward)
    
    def remove(self, receipt: Receipt, trial: TreeNode) -> None:
        "Remove the receipt and set this trial to be dead. "
        assert isinstance(trial, TreeNode), type(trial)
        assert trial.is_final(), "The removed trial should be a final node!"

        node, path = receipt
        self._decrement_virtual_loss(path, node)

        trail_node = self._treenode_store[trial.to_node()]
        trail_node.filtered = True

    # Here, the returned Any is just an element in the path. The type depends on how we build the search tree. If we follow the C++ implementation, it should be a kas_cpp_bindings.Next. If we use two-step generation for primitives, it should be either Union[primitive type, index of primitive], which is not yet implemented.
    def _ucd_select(self, node: TreeNode) -> Optional[Tuple[PseudoTreeNext, TreeNode]]:
        "Select a child of node, balancing exploration & exploitation"

        children = node.get_children(self._treenode_store)
        if len(children) == 0:
            logging.debug("Selection failed. ")
            return None

        # All children of node should already be expanded:
        assert len(node.get_unexpanded_children(self._treenode_store)) == 0

        N_children = [child.N for _, child in children]
        log_N_vertex = math.log(1 + sum(N_children))
        assert all([N > 0 for N in N_children])

        def ucd(key: Tuple[PseudoTreeNext, TreeNode]) -> float:
            """
            Upper confidence bound for trees.
            We save the tree as a DAG since multiple paths lead to a same node (by using different order of the primitives). Therefore, UCD (https://hal.science/hal-01499672/document) shall be used instead. 
            """
            _, child = key
            return child.Q / child.N + self._exploration_weight * math.sqrt(
                log_N_vertex / child.N
            ) - self.virtual_loss_constant * self.virtual_loss_count[child]

        selected_child = max(children, key=ucd)

        return selected_child

    def serialize(self) -> Dict:
        """Serialize the tree and return a dict."""

        node_list = []
        nodes = list(self._treenode_store.keys())

        # index each node
        index = {n: i for i, n in enumerate(nodes)}

        # dump father's n, q
        # dump children's n, q, filtered
        for i, n in enumerate(nodes):
            logging.debug(f"dumping node {i}")
            father = self._treenode_store[n]
            if father.N == 0:
                continue

            node_serial = {}
            node_serial['index'] = i
            node_serial['father'] = {
                'N': father.N,
                'Q': father.Q
            }
            node_serial['children'] = {}
            logging.debug("dumping children")
            for next, child in father.get_children(self._treenode_store):
                if child.N > 0 or child.is_final():
                    assert isinstance(next, Next.Type)
                    next_serial = self.next_serializer.serialize_type(next)
                    node_serial['children'][next_serial] = {
                        'N': child.N,
                        'Q': child.Q,
                        'children': [(n, index[c.to_node()]) for n, c in child.get_children(self._treenode_store) if c.N > 0]
                    }
                    if child.is_final():
                        node_serial['children'][next_serial]['filtered'] = child.filtered

            node_list.append(node_serial)

        packed_args = dict(
            virtual_loss_constant=self.virtual_loss_constant,
            leaf_num=self.leaf_num,
            exploration_weight=self._exploration_weight
        )
        j = dict(
            node_list=node_list,
            args=packed_args
        )
        return j
    
    def garbage_collect(self):
        """
        Remove non-root tree node with no predecessor.
        TODO
        """
        pass

    def _add_node(self, path: TreePath, node: Dict, node_factory: Dict) -> TreeNode:
        """Manually add tree nodes recursively. """
        _node = self._sampler.visit(path).to_node()
        tree_node = self._treenode_store[_node] if path.is_root() else TreeNode(_node)
        tree_node.N = node['father']['N']
        tree_node.Q = node['father']['Q']
        tree_node.children = []
        for _type_serial, child_serial in node['children'].items():
            _type = self.next_serializer.deserialize_type(_type_serial)
            child_path = path.concat(_type)
            child = TreeNode(_node, is_mid=True, type=_type)
            
            child.N = child_serial['N']
            child.Q = child_serial['Q']
            if child.is_final():
                child.filtered = child_serial['filtered']
            for next, grand_child_index in child_serial['children']:
                grand_child_path = child_path.concat(next)
                self._add_node(
                    grand_child_path, 
                    node_factory[grand_child_index], 
                    node_factory
                    )
                
            tree_node.children.append(child)
        if not path.is_root():
            self._treenode_store[_node] = tree_node
        return tree_node

    @staticmethod
    def deserialize(serialized: dict, sampler: Sampler) -> 'MCTS':
        """Deserialize a serialized tree and return a Tree object"""

        params = serialized['args']
        node_list = serialized['node_list']
        node_factory = {n['index']: n for n in node_list}
        tree = MCTS(sampler, **params)
        root_node = node_factory[0]
        tree._add_node(TreePath([]), root_node, node_factory)
        
        return tree
