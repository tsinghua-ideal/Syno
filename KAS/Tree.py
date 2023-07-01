import random
import math
import logging
from collections import defaultdict, OrderedDict
from typing import List, Tuple, Any, Optional, Dict, MutableSet

from .Node import Path, VisitedNode, Node
from .Sampler import Sampler
from .TreeNode import TreeNode, TreePath, PseudoTreeNext, PseudoArc
from .Bindings import Next, Arc
from .Utils import NextSerializer, AverageMeter


class MCTS:
    def __init__(self, sampler: Sampler, virtual_loss_constant: float = 0.0, leaf_num: int = 1, exploration_weight: float = math.sqrt(2), b: float=0.5, c_l: float=20) -> None:

        self._treenode_store: Dict[Node, TreeNode] = OrderedDict()
        self._root = sampler.visit([]).to_node()
        root_node = TreeNode(self._root)
        self._treenode_store[self._root] = root_node
        self.g_rave: Dict[PseudoArc, AverageMeter] = defaultdict(AverageMeter)

        self._sampler = sampler
        self._exploration_weight = exploration_weight
        self._c_l = c_l
        self._b = b
        random.seed(sampler._seed)

        # Tree Parallelization
        self.virtual_loss_constant = virtual_loss_constant
        self.virtual_loss_count: Dict[TreeNode, int] = defaultdict(
            int)  # node -> virtual loss count

        # Leaf Parallelization
        self.leaf_num = leaf_num
        
        self.next_serializer = NextSerializer()
        root_node.flush_T(1, self._treenode_store, self.g_rave, self._c_l, self._b)

    def _increment_virtual_loss(self, path: TreePath, node: Node, delta: int=1) -> None:
        assert delta > 0
        tree_node = self._treenode_store[node.to_node()]
        for next in path:
            tree_node = tree_node.get_child(next.type, self._treenode_store, on_tree=True)
            assert tree_node is not None
            self.virtual_loss_count[tree_node] += delta
            if next.key == 0:
                break
            tree_node = tree_node.get_child(next.key, self._treenode_store, on_tree=True)
            assert tree_node is not None
            self.virtual_loss_count[tree_node] += delta

    def _decrement_virtual_loss(self, path: TreePath, node: Node, delta: int=1) -> None:
        assert delta > 0
        tree_node = self._treenode_store[node.to_node()]
        for next in path:
            tree_node = tree_node.get_child(next.type, self._treenode_store, on_tree=True)
            assert tree_node is not None
            self.virtual_loss_count[tree_node] -= 1
            assert self.virtual_loss_count[tree_node] >= 0
            if next.key == 0:
                break
            tree_node = tree_node.get_child(next.key, self._treenode_store, on_tree=True)
            assert tree_node is not None
            self.virtual_loss_count[tree_node] -= 1
            assert self.virtual_loss_count[tree_node] >= 0

    # The receipt which is used for back propagation, which is the root node and the path.
    Receipt = Tuple[Node, TreePath]

    def do_rollout(self, node: VisitedNode) -> Optional[Tuple[Receipt, List[Tuple[TreePath, TreeNode]]]]:
        """
        Make the tree one layer better. (Train for one iteration.)

        Returns `((root, path), leaf)`. `path` is used to propagate reward. `leaf` is the result of simulation. Note that following `path` we do not necessarily arrive at `leaf`.
        
        If the tree is exhausted, then return None. 
        """
        while True:
            if self._treenode_store[node.to_node()].is_dead_end(self._treenode_store):
                logging.info("The tree is exhausted. ")
                return None
            result = self._may_fail_rollout(node)
            if result is not None:
                path, trials = result
                logging.debug(
                    f"Successful rollout: {path}. Evaluation to be done.")
                self._increment_virtual_loss(path, node, len(trials))
                return (node.to_node(), path), trials
            else:
                logging.debug(
                    f"During rollout, dead end encountered. Retrying...")

    def _may_fail_rollout(self, node: VisitedNode) -> Optional[Tuple[TreePath, List[Tuple[TreePath, TreeNode]]]]:
        "The trial may encounter a dead end. In this case, False is returned."
        tree_node = self._treenode_store[node.to_node()]

        # Select
        logging.debug("Selection start")
        select_result = self._select(tree_node)
        if select_result is None:
            return None
        path, leaf = select_result
        logging.debug(f"Selection end {path} {leaf}")

        # Expand
        if leaf.is_final():
            logging.debug("Selected final node, return immediately. ")
            return path, [(path, leaf) for _ in range(self.leaf_num)]
        if leaf.children_count(self._treenode_store, on_tree=True) == 0:
            if not leaf.is_fully_in_tree(self._treenode_store):
                leaf.add_new_children(self._treenode_store, self.g_rave, self._c_l)
                assert leaf.children_count(self._treenode_store, on_tree=True) > 0
            else:
                logging.debug(f"{leaf} is fully expanded and has no children.")
                return None
        
        logging.debug("Expansion start")
        expand_result = self._expand(leaf)
        if expand_result is None:
            return None
        next_expand, leaf_expanded = expand_result
        path = path.concat(next_expand)
        assert isinstance(path, TreePath), type(path)
        logging.debug(f"Expansion end {path} {leaf_expanded}")

        # Simulate
        leaves = []
        logging.debug(f"Simulation start")
        while not leaf_expanded.is_dead_end(self._treenode_store):
            leaf_simul = self._simulate(path, leaf_expanded)
            if leaf_simul is None: continue
            leaves.append(leaf_simul)
            if len(leaves) == self.leaf_num:
                break
        if leaf_expanded.is_dead_end(self._treenode_store):
            return None

        return path, leaves
    
    #########################
    #   Functional support
    #########################

    def _select(self, node: TreeNode) -> Optional[Tuple[TreePath, TreeNode]]:
        "Find an unexplored descendent of `node`"
        # Here, the path is just arbitrary, and depends on how we build the search tree. See doc for `_uct_select`. We only need to make sure we can construct a `TwoStepPath` from `path`.
        path = TreePath([])
        while True:
            if node.is_terminal(self._treenode_store) or \
                len(node.get_unexpanded_children(self._treenode_store, on_tree=True)) > 0 or \
                    node.children_count(self._treenode_store, on_tree=True) == 0:
                # node is terminal, unexplored, or a leaf
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

        unexpanded_children = node.get_unexpanded_children(self._treenode_store, on_tree=True)
        if len(unexpanded_children) == 0:
            logging.debug("Expand failed.")
            return None

        # randomly select a child from pool
        next, leaf = random.choice(unexpanded_children)
        return next, leaf

    def _simulate(self, path: TreePath, node: TreeNode) -> Optional[Tuple[TreePath, TreeNode]]:
        "Returns a random simulation (to completion) of `node`"

        def random_child(node: TreeNode) -> Optional[Tuple[TreePath, TreeNode, AverageMeter]]:
            """
            Two step random selection. First, randomly select a primitive type. Then, randomly select a child of that type.
            """
            assert isinstance(node, TreeNode)
            children = node.get_children(self._treenode_store)
            assert all([not c.is_dead_end(self._treenode_store) for _, c, _ in children])
            # logging.debug(f"{node} has children {children}")
            if len(children) == 0:
                return None
            next, selected_child, edge_state = random.choice(children)
            # logging.debug(f"Random selected {next}: {selected_child}")
            return next, selected_child, edge_state

        while not node.is_terminal(self._treenode_store):
            random_select_result = random_child(node)
            if random_select_result is None:
                logging.debug(f"Simulation Encountered dead father {node}")
                assert node.is_dead_end(self._treenode_store)
                return None
            next, node, edge_state = random_select_result
            path = path.concat(next)
        if node.is_dead_end(self._treenode_store): # is dead end
            logging.debug(f"Simulation Encountered dead end {node}")
            return None
        assert node.is_final()
        return path, node

    def back_propagate(self, receipt: Receipt, reward: float, path_to_trail: TreePath) -> None:
        """
        Send the reward back up to the ancestors of the leaf
        """
        assert isinstance(reward, float)
        assert isinstance(path_to_trail, TreePath)
        assert 0.0 <= reward <= 1.0

        node, path = receipt
        self._decrement_virtual_loss(path, node)
        
        def update_stats(node: TreeNode, reward: float, arc: PseudoArc = None) -> None:
            node.update(reward, arc)
            node.flush_T(node.state.N, self._treenode_store, self.g_rave, self._c_l, self._b)
        
        node = self._treenode_store[node.to_node()]
        for next in path:
            arc = node._node.get_arc_from_handle(next)
            assert arc is not None or next.key == 0
            update_stats(node, reward, next.type)
            node = node.get_child(next.type, self._treenode_store, on_tree=True)
            assert node is not None
            update_stats(node, reward, arc)
            if next.key == 0:
                break
            node = node.get_child(next.key, self._treenode_store, on_tree=True)
            assert node is not None
            
        update_stats(node, reward)
            
        # Update g rave
        arcs = node._node.get_composing_arcs()
        for arc in arcs:
            self.g_rave[arc].update(reward)
        update_types = set([next.type for next in path_to_trail])
        for type in update_types:
            self.g_rave[type].update(reward)
    
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
        """
        Select a child of node, balancing exploration & exploitation
        """

        children = node.get_children(self._treenode_store, on_tree=True)
        if len(children) == 0:
            # node.add_new_children(self._treenode_store, self.g_rave, self._c_l)
            logging.debug("Selection failed. ")
            return None

        # All children of node should already be expanded:
        assert len(node.get_unexpanded_children(self._treenode_store, on_tree=True)) == 0

        log_N_vertex = math.log(node.state.N)

        def ucb1_tuned(key: Tuple[PseudoTreeNext, TreeNode, AverageMeter]) -> float:
            """
            Upper confidence bound for trees.
            We save the tree as a DAG since multiple paths lead to a same node (by using different order of the primitives). Therefore, UCD (https://hal.science/hal-01499672/document) shall be used instead. (Now we use only a simple version of UCD, which is not the same as the one in the paper.)
            TOTST: Implement the UCB1-tuned. 
            """
            _, child, edge = key
            return edge.mean + self._exploration_weight * math.sqrt(
                (log_N_vertex / edge.N) * min(0.25, edge.std * edge.std + math.sqrt(2 * log_N_vertex / edge.N))
            ) - self.virtual_loss_constant * self.virtual_loss_count[child]

        selected_child = max(children, key=ucb1_tuned)

        return selected_child

    ##########################
    #      I/O support
    ##########################
    
    def serialize(self) -> Dict:
        """Serialize the tree and return a dict."""

        self.garbage_collect(keep_tree_node=False)
        
        node_list = []
        nodes = list(self._treenode_store.keys())

        # index each node
        index = {n: i for i, n in enumerate(nodes)}

        # dump father"s n, q
        # dump children"s n, q, filtered
        for i, n in enumerate(nodes):
            logging.debug(f"dumping node {i}")
            father = self._treenode_store[n]

            node_serial = {}
            node_serial["index"] = i
            node_serial["father"] = {}
            node_serial["father"]["state"] = father.state_dict
            if father.is_final():
                node_serial["father"]["filtered"] = father.filtered
                node_serial["father"]["reward"] = father.reward
            node_serial["children"] = {}
            logging.debug("dumping children")
            for next, child, _ in father.get_children(self._treenode_store, auto_initialize=False):
                assert isinstance(next, Next.Type)
                next_serial = self.next_serializer.serialize_type(next)
                children = [
                    (n, index[c.to_node()], e) 
                        for n, c, e in child.get_children(self._treenode_store, auto_initialize=False) 
                        if c._node in index
                ]
                children_with_rave_score: List[Tuple[PseudoTreeNext, int, Dict[str, Dict]]] = []
                for n, ind, e in children:
                    arc = child._node.get_arc_from_handle(Next(next, n))
                    assert arc is not None
                    lrave = child.l_rave[arc].serialize()
                    grave = self.g_rave[arc].serialize()
                    rave_score = {
                        "lrave": lrave,
                        "grave": grave
                    }
                    children_with_rave_score.append((n, ind, e.serialize(), rave_score))
                node_serial["children"][next_serial] = {
                        "state": child.state_dict, 
                        "lrave": father.l_rave[next].serialize(),
                        "grave": self.g_rave[next].serialize(),
                        "children": children_with_rave_score
                    }

            node_list.append(node_serial)

        packed_args = dict(
            virtual_loss_constant=self.virtual_loss_constant,
            leaf_num=self.leaf_num,
            exploration_weight=self._exploration_weight,
            b=self._b,
            c_l=self._c_l
        )
        j = dict(
            node_list=node_list,
            args=packed_args
        )
        return j
    

    def _add_node(self, path: TreePath, node: Dict, node_factory: Dict) -> TreeNode:
        """Manually add tree nodes recursively. """
        node_visited = self._sampler.visit(path)
        if node_visited is None:
            return
        
        node_ = node_visited.to_node()
        tree_node = self._treenode_store[node_] if path.is_root() else TreeNode(node_)
        tree_node.load(node["father"]["state"])
        tree_node.children = []
        if tree_node.is_final():
            tree_node.reward = node["father"]["reward"]
            tree_node.filtered = node["father"]["filtered"]
        for _type_serial, child_serial in node["children"].items():
            _type = self.next_serializer.deserialize_type(_type_serial)
            child_path = path.concat(_type)
            child = TreeNode(node_, is_mid=True, type=_type)
            
            # Rave
            tree_node.l_rave[_type] = AverageMeter.deserialize(child_serial["lrave"])
            g_rave_am = AverageMeter.deserialize(child_serial["grave"])
            if not self.g_rave[_type].empty():
                assert g_rave_am == self.g_rave[_type], "g_rave inconsistency found!"
            else:
                self.g_rave[_type] = g_rave_am
            
            child.load(child_serial["state"])
            for next, grand_child_index, edge_serial, rave_score in child_serial["children"]:     
                grand_child_path = child_path.concat(next)
                child_node = self._add_node(
                    grand_child_path, 
                    node_factory[grand_child_index], 
                    node_factory
                    )
                
                grand_child_next = grand_child_path[-1]
                assert grand_child_next == Next(_type, next), (grand_child_next, Next(_type, next))
                grand_child_arc = child._node.get_arc_from_handle(grand_child_next)
                assert grand_child_arc is not None
                child_node.edge_states[grand_child_next.key] = AverageMeter.deserialize(edge_serial)
                # Rave
                child_node.l_rave[grand_child_arc] = AverageMeter.deserialize(rave_score["lrave"])
                g_rave_am = AverageMeter.deserialize(rave_score["grave"])
                if grand_child_arc in self.g_rave.keys():
                    assert g_rave_am == self.g_rave[grand_child_arc], "g_rave inconsistency found!"
                else:
                    self.g_rave[grand_child_arc] = g_rave_am
                
            tree_node.children.append(child)
        if not path.is_root():
            self._treenode_store[node_] = tree_node
        return tree_node

    @staticmethod
    def deserialize(serialized: dict, sampler: Sampler) -> "MCTS":
        """Deserialize a serialized tree and return a Tree object"""

        params = serialized["args"]
        node_list = serialized["node_list"]
        node_factory = {n["index"]: n for n in node_list}
        tree = MCTS(sampler, **params)
        root_node = node_factory[0]
        tree._add_node(TreePath([]), root_node, node_factory)
        
        return tree

    def garbage_collect(self, keep_tree_node:bool=True):
        """
        Remove non-root tree node with no predecessor. 
        All nodes that are accessible from the root should be reserved! 
        """
        # Label every alive node
        alive_nodes: MutableSet[Node] = set()
        def label_alive(node: TreeNode) -> bool:
            """Return a boolean value indicating whether the node contains a final descendant. """
            assert isinstance(node, TreeNode)
            children = node.get_children(self._treenode_store, False)
            alive_flag = node.N > 0 or node.is_final() or (keep_tree_node and node._isin_tree) or node.is_dead_end(self._treenode_store)
            if not node.is_dead_end(self._treenode_store):
                for _, child, _ in children:
                    child_alive_flag = label_alive(child)
                    alive_flag = alive_flag or child_alive_flag
            if alive_flag:
                alive_nodes.add(node.to_node())
            return alive_flag
        assert label_alive(self._treenode_store[self._root])
        
        # Remove dead nodes
        key_list = list(self._treenode_store.keys())
        for node in key_list:
            if node not in alive_nodes and not node.is_final():
                self._treenode_store.pop(node)