import random
import math
import logging
from collections import defaultdict, OrderedDict as ODict
from typing import List, Tuple, Any, Optional, DefaultDict, Set, Union, OrderedDict, Dict
from copy import deepcopy

from .Node import Path, VisitedNode, Node
from .Sampler import Sampler
from .TreeNode import TreeNode, TreePath, PseudoTreeNext, PseudoArc
from .Bindings import Next, Arc
from .Utils import NextSerializer, AverageMeter

MockArc = Union[Next, Arc]

class MCTS:
    def __init__(self, sampler: Sampler, virtual_loss_constant: float = 0.0, leaf_num: int = 1, exploration_weight: float = math.sqrt(2), b: float=0.5, c_l: float=20, policy: str='update-descent', continue_after_exhaust: bool=False) -> None:

        self._treenode_store: OrderedDict[Node, TreeNode] = ODict()
        self._root = sampler.visit([]).to_node()
        self.tree_root = TreeNode(self._root)
        self._treenode_store[self._root] = self.tree_root
        self.g_rave: DefaultDict[PseudoArc, AverageMeter] = defaultdict(AverageMeter)

        self._sampler = sampler
        self._exploration_weight = exploration_weight
        self._c_l = c_l
        self._b = b
        self._policy = policy
        self.continue_after_exhaust = continue_after_exhaust
        assert policy in ['update-all', 'update-descent']
        random.seed(sampler._seed)

        # Tree Parallelization
        self.virtual_loss_constant = virtual_loss_constant
        self.virtual_loss_count: DefaultDict[TreeNode, int] = defaultdict(
            int)  # node -> virtual loss count

        # Leaf Parallelization
        self.leaf_num = leaf_num
        
        self.next_serializer = NextSerializer()
        self.tree_root.flush_T(1, self._treenode_store, self.g_rave, self._c_l, self._b, filter_exhausted=not self.continue_after_exhaust)

    def _increment_virtual_loss(self, path: TreePath, node: Node, delta: int=1) -> None:
        assert delta > 0
        tree_node = self._treenode_store[node.to_node()]
        for next in path:
            tree_node, _ = tree_node.get_child(next.type, self._treenode_store, on_tree=True)
            assert tree_node is not None
            self.virtual_loss_count[tree_node] += delta
            if next.key == 0:
                break
            tree_node, _ = tree_node.get_child(next.key, self._treenode_store, on_tree=True)
            assert tree_node is not None
            self.virtual_loss_count[tree_node] += delta

    def _decrement_virtual_loss(self, path: TreePath, node: Node, delta: int=1) -> None:
        assert delta > 0
        tree_node = self._treenode_store[node.to_node()]
        for next in path:
            tree_node, _ = tree_node.get_child(next.type, self._treenode_store, on_tree=True)
            assert tree_node is not None
            self.virtual_loss_count[tree_node] -= 1
            assert self.virtual_loss_count[tree_node] >= 0
            if next.key == 0:
                break
            tree_node, _ = tree_node.get_child(next.key, self._treenode_store, on_tree=True)
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
        continue_after_exhausted is an option if you want to rollout even after the tree is exhausted. (Thus you may get a better estimate of the value of the root node.)
        """
        while True:
            if not self.continue_after_exhaust and self.tree_root.is_exhausted(self._treenode_store):
                logging.info("The tree is exhausted. ")
                return None
            result = self._may_fail_rollout(node)
            if result is not None:
                path, trials = result
                logging.debug(
                    f"Successful rollout: {path} {trials}. Evaluation to be done.")
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
        if leaf.children_count(self._treenode_store, on_tree=True, filter_exhausted=not self.continue_after_exhaust) == 0:
            if not leaf.is_fully_in_tree(self._treenode_store):
                leaf.add_new_children(self._treenode_store, self.g_rave, self._c_l)
                assert leaf.children_count(self._treenode_store, on_tree=True) > 0
            else:
                logging.debug(f"{leaf} is fully expanded and has no children.")
                return None
        
        logging.debug("Expansion start")
        # leaf_expanded = leaf
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
        leaf_expanded._node.expand_async(3)
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
            next, node, _ = selected
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
        next, leaf, _ = random.choice(unexpanded_children)
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

    def back_propagate(self, receipt: Receipt, reward: float, path_to_trial: TreePath) -> None:
        """
        Send the reward back up to the ancestors of the leaf
        """
        
        # check input types
        assert isinstance(reward, float)
        assert isinstance(path_to_trial, TreePath)
        assert 0.0 <= reward <= 1.0

        node, path = receipt
        self._decrement_virtual_loss(path, node)
        
        # update rewards
        leaf_node = self.update_nodes(receipt, reward)
        assert not leaf_node.is_dead_end(self._treenode_store), leaf_node._node.get_possible_path()
        
        # update rave scores
        arcs = leaf_node._node.get_composing_arcs()
        update_types = set([next.type for next in path_to_trial])
            
        self.update_grave(arcs, update_types, reward)
        self.update_lrave(leaf_node, arcs, reward)
    
    def update_nodes(self, receipt: Receipt, reward: float) -> TreeNode:
        node, path = receipt
        
        def update_stats(tree_node: TreeNode, reward: float, arc: Optional[PseudoArc] = None) -> None:
            tree_node.update(reward, arc)
            tree_node.flush_T(tree_node.N, self._treenode_store, self.g_rave, self._c_l, self._b, filter_exhausted=not self.continue_after_exhaust)

        tree_node = self._treenode_store[node.to_node()]
        if self._policy == 'update-descent':
            for tree_next in path:
                arc = tree_node._node.get_arc_from_handle(tree_next)
                # TODO: typing check error in Python 3.8
                # assert isinstance(arc, MockArc) or tree_next.key == 0, f"{arc, tree_next}"
                update_stats(tree_node, reward, tree_next.type)
                tree_node, _ = tree_node.get_child(tree_next.type, self._treenode_store, on_tree=True)
                assert tree_node is not None
                if tree_next.key == 0:
                    break
                update_stats(tree_node, reward, arc)
                tree_node, _ = tree_node.get_child(tree_next.key, self._treenode_store, on_tree=True)
                assert tree_node is not None
                
            update_stats(tree_node, reward)
            final_node = tree_node
        elif self._policy == 'update-all':
            logging.warning("Update-all policy is still not fully implemented. Please do not use it for now!")
            for tree_next in path:
                arc = tree_node._node.get_arc_from_handle(tree_next)
                tree_node, _ = tree_node.get_child(tree_next.type, self._treenode_store, on_tree=True)
                if tree_next.key == 0:
                    break
                tree_node, _ = tree_node.get_child(tree_next.key, self._treenode_store, on_tree=True)
            assert tree_node is not None
            final_node = tree_node
            logging.debug(f"final node is {final_node}")
            arcs = final_node._node.get_composing_arcs()
            
            # Mark Ancestors
            ancestors: List[TreeNode] = []
            def mark_ancestors(src_node: TreeNode, tgt_node: TreeNode, arc_pool: Set[Arc]) -> bool:
                logging.debug(f"mark_ancestors({src_node}, {tgt_node}, {arc_pool})")
                if src_node == tgt_node or tgt_node in src_node.children:
                    return True
                
                updated_flag = False
                for arc in arc_pool:
                    if src_node._node.can_accept_arc(arc):
                        arc_pool.remove(arc)
                        nxt = arc.to_next()
                        mid_child = src_node.get_child(nxt.type, self._treenode_store)[0]
                        assert mid_child is not None
                        child_node = self.touch(src_node._node.get_child_from_arc(arc))
                        if mark_ancestors(child_node, tgt_node, arc_pool) and not updated_flag:
                            ancestors.append(child_node)
                            updated_flag = True
                        arc_pool.add(arc)
                return updated_flag
            assert mark_ancestors(self.tree_root, final_node, set(arcs))
            ancestors = set(ancestors)
            
            # Compute Weights
            node_weights: DefaultDict[TreeNode, float] = defaultdict(float)
            edge_buffer: DefaultDict[TreeNode, Set[int]] = defaultdict(set)
            node_weights[self.tree_root] = 1.0
            node_weights[final_node] = 1.0
            
            def compute_weights(src_node: TreeNode, tgt_node: TreeNode, arc_pool: Set[Arc], current_weight: float, alpha: float) -> bool:
                logging.debug(f"compute_weights({src_node}, {tgt_node}, {arc_pool}, {current_weight})")
                if src_node == tgt_node or tgt_node in src_node.children:
                    return True
                
                updated_flag = False
                potential_children: Dict[TreeNode, Set[TreeNode]] = defaultdict(set)
                
                for arc in arc_pool:
                    if src_node._node.can_accept_arc(arc):
                        arc_pool.remove(arc)
                        nxt = arc.to_next()
                        mid_child = src_node.get_child(nxt.type, self._treenode_store)[0]
                        assert mid_child is not None
                        child_node = self.touch(src_node._node.get_child_from_arc(arc))
                        if child_node in ancestors and not updated_flag:
                            potential_children[mid_child].add(child_node)
                            edge_buffer[mid_child].add(nxt.key)
                            updated_flag = True
                        arc_pool.add(arc)
                        
                if updated_flag:
                    mid_child_weight: float = current_weight / len(potential_children.keys())
                    for child_node, child_nodes in potential_children.items():
                        node_weights[child_node] += mid_child_weight
                        for child_node in child_nodes:
                            child_weight = mid_child_weight / len(child_nodes)
                            node_weights[child_node] += child_weight
                            logging.debug(f"node_weights[{child_node}] += {child_weight}")
                
                return updated_flag
            
            queue = [self.tree_root]
            while len(queue) > 0:
                node = queue.pop(0)
                compute_weights(node, final_node, set(arcs), node_weights[node], 1.0)
            logging.debug(f"weights computed. node_weights: {node_weights}, edge_buffer: {edge_buffer}")
            
            # Update
            for tree_node_toupd, weight in node_weights.items():
                update_stats(tree_node_toupd, reward * weight)
                if tree_node_toupd._is_mid:
                    for key in edge_buffer[tree_node_toupd]:
                        tree_node_toupd.update_edge(reward * weight / len(edge_buffer[tree_node_toupd]), key)
        
        return final_node
    
    def update_grave(self, arcs: Set[Arc], update_types: Set[Next.Type], reward: float) -> None:
        for arc in arcs:
            self.g_rave[arc].update(reward)
        for type in update_types:
            self.g_rave[type].update(reward)
    
    def update_lrave(self, final_node: TreeNode, arcs: Set[Arc], reward: float) -> None:
        def attempt_to_node(src_node: TreeNode, tgt_node: TreeNode, arc_pool: Set[Arc]) -> bool:
            """
            Attempt to reach tgt_node from src_node with arcs in arc_pool
            """
            if src_node == tgt_node:
                return True
            
            updated_flag = False  # L-Rave should be updated at most once
            
            for arc in list(arc_pool):
                if src_node._node.can_accept_arc(arc):
                    arc_pool.remove(arc)
                    nxt = arc.to_next()
                    mid_child = src_node.get_child(nxt.type, self._treenode_store)
                    if mid_child is None: 
                        continue
                    mid_child = mid_child[0]
                    child_node = self.touch(src_node._node.get_child_from_arc(arc))
                    if child_node.is_dead_end(self._treenode_store): 
                        continue
                    if attempt_to_node(child_node, tgt_node, arc_pool) and not updated_flag:
                        src_node.update_lrave(reward, nxt.type)
                        mid_child.update_lrave(reward, arc)
                        updated_flag = True
                    arc_pool.add(arc)
            
            return updated_flag
                
        attempt_to_node(self.tree_root, final_node, set(arcs))
    
    def touch(self, node: Node) -> TreeNode:
        """
        Initialize node if not exist in store
        """
        assert isinstance(node, Node)
        if node not in self._treenode_store.keys():
            self._treenode_store[node] = TreeNode(node)
        return self._treenode_store[node]
    
    def remove(self, receipt: Receipt, trial: TreeNode) -> None:
        "Remove the receipt and set this trial to be dead. "
        assert isinstance(trial, TreeNode), type(trial)
        assert trial.is_final(), "The removed trial should be a final node!"

        node, path = receipt
        self._decrement_virtual_loss(path, node)

        trail_node = self._treenode_store[trial.to_node()]
        trail_node.filtered = True

        tree_node = self._treenode_store[node.to_node()]
        flush_list: List[TreeNode] = []
        for next in path:
            child = tree_node.get_child(next.type, self._treenode_store, on_tree=True)
            if child is None:
                break
            tree_node = child[0]
            flush_list.append(tree_node)
            if next.key == 0:
                break
            child = tree_node.get_child(next.key, self._treenode_store, on_tree=True)
            if child is None:
                break
            tree_node = child[0]
            flush_list.append(tree_node)
        
        for tree_node in flush_list[::-1]:
            tree_node.flush_T(tree_node._last_T, self._treenode_store, self.g_rave, self._c_l, self._b, filter_exhausted=not self.continue_after_exhaust)

    # Here, the returned Any is just an element in the path. The type depends on how we build the search tree. If we follow the C++ implementation, it should be a kas_cpp_bindings.Next. If we use two-step generation for primitives, it should be either Union[primitive type, index of primitive], which is not yet implemented.
    def _ucd_select(self, node: TreeNode) -> Optional[Tuple[PseudoTreeNext, TreeNode, AverageMeter]]:
        """
        Select a child of node, balancing exploration & exploitation
        """

        children = node.get_children(self._treenode_store, on_tree=True)
        
        if not self.continue_after_exhaust:
            # filter exhausted children
            children = [child for child in children if not child[1].is_exhausted(self._treenode_store)]
            
        if len(children) == 0:
            # node.add_new_children(self._treenode_store, self.g_rave, self._c_l)
            assert node.is_exhausted(self._treenode_store), f"The node {node} should be exhausted if it has no children. "
            logging.debug("Selection failed. ")
            return None

        # All children of node should already be expanded:
        assert len(node.get_unexpanded_children(self._treenode_store, on_tree=True)) == 0

        log_N_vertex = math.log(node.N)

        def ucb1_tuned(key: Tuple[PseudoTreeNext, TreeNode, AverageMeter]) -> float:
            """
            Upper confidence bound for trees.
            We save the tree as a DAG since multiple paths lead to a same node (by using different order of the primitives). Therefore, UCD (https://hal.science/hal-01499672/document) shall be used instead. (Now we use only a simple version of UCD, which is not the same as the one in the paper.)
            TOTST: Implement the UCB1-tuned. 
            """
            _, child, edge = key
            if edge.N == 0:
                return math.inf
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
                    children_with_rave_score.append([n, ind, e.serialize(), rave_score])
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
            c_l=self._c_l,
            continue_after_exhaust=self.continue_after_exhaust,
        )
        j = dict(
            node_list=node_list,
            args=packed_args
        )
        return j

    @staticmethod
    def deserialize(serialized: dict, sampler: Sampler) -> "MCTS":
        """Deserialize a serialized tree and return a Tree object"""
        
        def _add_node(mcts: MCTS, path: TreePath, node: Dict, node_factory: Dict) -> TreeNode:
            """Manually add tree nodes recursively. """
            node_visited = mcts._sampler.visit(path)
            if node_visited is None:
                return
            
            node_ = node_visited.to_node()
            tree_node = mcts._treenode_store[node_] if path.is_root() else TreeNode(node_)
            tree_node.load(node["father"]["state"])
            tree_node.children = []
            if tree_node.is_final():
                tree_node.reward = node["father"]["reward"]
                tree_node.filtered = node["father"]["filtered"]
            for _type_serial, child_serial in node["children"].items():
                _type = mcts.next_serializer.deserialize_type(_type_serial)
                child_path = path.concat(_type)
                child_node = TreeNode(node_, is_mid=True, type=_type)
                
                # Rave
                tree_node.l_rave[_type].load(child_serial["lrave"])
                mcts.g_rave[_type].load(child_serial["grave"])
                
                child_node.load(child_serial["state"])
                for next, grand_child_index, edge_serial, rave_score in child_serial["children"]:     
                    grand_child_path = child_path.concat(next)
                    grand_child_node = _add_node(
                        mcts,
                        grand_child_path, 
                        node_factory[grand_child_index], 
                        node_factory
                        )
                    
                    grand_child_next = grand_child_path[-1]
                    assert grand_child_next == Next(_type, next), (grand_child_next, Next(_type, next))
                    grand_child_arc = child_node._node.get_arc_from_handle(grand_child_next)
                    assert grand_child_arc is not None
                    child_node.edge_states[grand_child_next.key].load(edge_serial)
                    # Rave
                    child_node.l_rave[grand_child_arc].load(rave_score["lrave"])
                    mcts.g_rave[grand_child_arc].load(rave_score["grave"])
                    
                tree_node.children.append(child_node)
            if not path.is_root():
                mcts._treenode_store[node_] = tree_node
            return tree_node

        params = serialized["args"]
        node_list = serialized["node_list"]
        node_factory = {n["index"]: n for n in node_list}
        tree = MCTS(sampler, **params)
        root_node = node_factory[0]
        _add_node(tree, TreePath([]), root_node, node_factory)
        
        return tree

    def garbage_collect(self, keep_tree_node:bool=True):
        """
        Remove non-root tree node with no predecessor. 
        All nodes that are accessible from the root should be reserved! 
        """
        # Label every alive node
        alive_nodes: Set[Node] = set()
        def label_alive(node: TreeNode) -> bool:
            """Return a boolean value indicating whether the node contains a final descendant. """
            assert isinstance(node, TreeNode)
            children = node.get_children(self._treenode_store, False)
            alive_flag = not node.empty() or (keep_tree_node and node._isin_tree)
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
        
        # Clean RAVE dict
        for k, rave_score in list(self.g_rave.items()):
            if rave_score.empty():
                self.g_rave.pop(k)
                
        for node, tree_node in self._treenode_store.items():
            for k, rave_score in list(tree_node.l_rave.items()):
                if rave_score.empty():
                    tree_node.l_rave.pop(k)