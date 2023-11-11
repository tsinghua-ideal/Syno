import random
import math
import logging
import time
import threading, queue
from collections import defaultdict
from typing import List, Tuple, Optional, DefaultDict, Set, Union, Dict
from multiset import Multiset

from KAS import Node, Path, Sampler, Next, Arc, NextSerializer

from .node import TreeNode, TreePath, PseudoTreeNext, PseudoArc
from .avg_meter import AverageMeter


MockArc = Union[Next, Arc]


class MCTSTree:
    def __init__(
        self,
        sampler: Sampler,
        virtual_loss_constant: float = 0.0,
        leaf_num: int = 1,
        exploration_weight: float = math.sqrt(2),
        b: float = 0.5,
        c_l: float = 20,
        policy: str = "update-descent",
        max_final_iterations: Tuple[int, int, int] = (
            10,
            1.5,
            3000,
        ),  # (a, b, c) s.t. sample num = a * (b ** (max_depth - depth))
        simulate_retry_period: float = 60,
        sample_retry_times: int = 5,
        simulate_decay_time: Tuple[float, float] = (300, 300),
        max_depth: int = 16,
        rave_random_ratio: float = 0.3,
    ) -> None:
        self._treenode_store: Dict[Node, TreeNode] = dict()
        self._path_store: Dict[Node, Path] = dict()
        self._root = sampler.root().to_node()
        assert not self._root.is_dead_end()
        self.tree_root = self.touch(self._root, path=Path([]))
        self.g_rave: DefaultDict[PseudoArc, AverageMeter] = defaultdict(AverageMeter)

        self._sampler = sampler
        self._exploration_weight = exploration_weight
        self._c_l = c_l
        self._b = b
        self._policy = policy
        self._max_final_iterations = max_final_iterations
        self._simulate_retry_period = simulate_retry_period
        self._sample_retry_times = sample_retry_times
        self._max_depth = max_depth
        self._rave_random_ratio = rave_random_ratio
        assert 0 <= rave_random_ratio <= 1
        random.seed(sampler._seed)

        # Tree Parallelization
        self.virtual_loss_constant = virtual_loss_constant

        # Leaf Parallelization
        self.leaf_num = leaf_num

        self.next_serializer = NextSerializer()

        self.simulate_time_ema = 0
        self.simulate_decay_time = simulate_decay_time
        self.stuck_path = None

        # HACK
        tree_node = self.tree_root
        while True:
            tree_node.state.N = 1
            tree_node._isin_tree = True
            tree_node._last_T = 1
            if tree_node.children_count() == 1:
                tree_node.reveal_new_children()
                tree_node = tree_node.get_children(on_tree=True)[0][1]
            else:
                break

    def _increment_virtual_loss(self, path: TreePath, delta: int = 1) -> None:
        assert delta > 0
        # logging.debug(f"increment virtual loss by {delta} of {path}")
        tree_node = self.tree_root
        tree_node._virtual_loss += delta
        # logging.debug(
        #     f"increment virtual loss by {delta} at {tree_node}, virtual loss count: {tree_node._virtual_loss - delta} -> {tree_node._virtual_loss}"
        # )
        for next in path:
            tree_node = tree_node.get_child(next.type, on_tree=True)
            if tree_node is None:
                # logging.debug(f"stopped at {next.type}")
                break
            tree_node, _ = tree_node
            tree_node._virtual_loss += delta
            # logging.debug(
            #     f"increment virtual loss by {delta} at {tree_node}, virtual loss count: {tree_node._virtual_loss - delta} -> {tree_node._virtual_loss}"
            # )
            if next.key == 0:
                break
            tree_node = tree_node.get_child(next.key, on_tree=True)
            if tree_node is None:
                # logging.debug(f"stopped at {next.key}")
                break
            tree_node, _ = tree_node
            tree_node._virtual_loss += delta
            # logging.debug(
            #     f"increment virtual loss by {delta} at {tree_node}, virtual loss count: {tree_node._virtual_loss - delta} -> {tree_node._virtual_loss}"
            # )

    def _decrement_virtual_loss(self, path: TreePath, delta: int = 1) -> None:
        assert delta > 0
        # logging.debug(f"decrement virtual loss by {delta} of {path}")
        tree_node = self.tree_root
        tree_node._virtual_loss -= delta
        # logging.debug(
        #     f"decrement virtual loss by {delta} at {tree_node}, virtual loss count: {tree_node._virtual_loss + delta} -> {tree_node._virtual_loss}"
        # )
        if tree_node._virtual_loss < 0:
            tree_node._virtual_loss = 0
            logging.warning("Virtual loss go below 0! ")
        for next in path:
            tree_node = tree_node.get_child(next.type, on_tree=True)
            # assert tree_node is not None
            if tree_node is None:
                # logging.debug(f"stopped at {next.type}")
                break
            tree_node, _ = tree_node
            tree_node._virtual_loss -= delta
            # logging.debug(
            #     f"decrement virtual loss by {delta} at {tree_node}, virtual loss count: {tree_node._virtual_loss + delta} -> {tree_node._virtual_loss}"
            # )
            if tree_node._virtual_loss < 0:
                tree_node._virtual_loss = 0
                logging.warning("Virtual loss go below 0! ")
            if next.key == 0:
                break
            tree_node = tree_node.get_child(next.key, on_tree=True)
            # assert tree_node is not None
            if tree_node is None:
                # logging.debug(f"stopped at {next.key}")
                break
            tree_node, _ = tree_node
            tree_node._virtual_loss -= delta
            # logging.debug(
            #     f"decrement virtual loss by {delta} at {tree_node}, virtual loss count: {tree_node._virtual_loss + delta} -> {tree_node._virtual_loss}"
            # )
            if tree_node._virtual_loss < 0:
                tree_node._virtual_loss = 0
                logging.warning("Virtual loss go below 0! ")

    def _resurrect(self, path: TreePath) -> None:
        tree_node = self.tree_root
        for next in path:
            tree_node = tree_node.get_child(
                next.type, on_tree=True, include_simulate_failure=False
            )
            if tree_node is None:
                break
            tree_node, _ = tree_node
            if tree_node._simulate_fail:
                logging.debug(f"Resurrected {tree_node} during lrave update. ")
                tree_node._simulate_fail = False
            if next.key == 0:
                break
            tree_node = tree_node.get_child(
                next.key, on_tree=True, include_simulate_failure=False
            )
            if tree_node is None:
                break
            tree_node, _ = tree_node
            if tree_node._simulate_fail:
                logging.debug(f"Resurrected {tree_node} during lrave update. ")
                tree_node._simulate_fail = False

    def visit(
        self, path: TreePath, on_tree: bool = True, put_in_tree: bool = False
    ) -> Optional[TreeNode]:
        tree_node = self.tree_root
        if put_in_tree:
            tree_node._isin_tree = True
        for next in path:
            tree_node = tree_node.get_child(
                next.type, auto_initialize=not on_tree, on_tree=on_tree
            )
            if tree_node is None:
                return None
            tree_node, _ = tree_node
            if put_in_tree:
                tree_node._isin_tree = True
            if next.key == 0:
                break
            tree_node = tree_node.get_child(
                next.key, auto_initialize=not on_tree, on_tree=on_tree
            )
            if tree_node is None:
                return None
            tree_node, _ = tree_node
            if put_in_tree:
                tree_node._isin_tree = True
        return tree_node

    # The receipt which is used for back propagation, which is the root node and the path.
    Receipt = TreePath

    def do_rollout(
        self, check_exhaustion: bool = True
    ) -> Optional[Tuple[Receipt, List[Tuple[TreePath, TreeNode]]]]:
        """
        Make the tree one layer better. (Train for one iteration.)

        Returns `((root, path), leaf)`. `path` is used to propagate reward. `leaf` is the result of simulation. Note that following `path` we do not necessarily arrive at `leaf`.

        If the tree is exhausted, then return None.
        """
        while True:
            if check_exhaustion and self.tree_root.is_exhausted():
                logging.info("The tree is exhausted. ")
                return None
            if self.stuck_path is not None:
                logging.info(f"The tree has stuck at {self.stuck_path}. ")
                return None
            start = time.time()
            result = self._may_fail_rollout()
            logging.debug(f"Time elapsed {time.time() - start}")

            if result is not None:
                path, trials = result
                logging.debug(
                    f"Successful rollout: {path} {trials}. Evaluation to be done."
                )
                assert all(trial.is_final() for _, trial in trials)
                assert len(trials) == self.leaf_num
                return path, trials
            else:
                logging.debug(f"During rollout, dead end encountered. Retrying...")

    def _may_fail_rollout(
        self,
    ) -> Optional[Tuple[TreePath, List[Tuple[TreePath, TreeNode]]]]:
        "The trial may encounter a dead end. In this case, False is returned."

        # Select
        logging.debug("Selection start")
        select_result = self.select()
        if select_result is None:
            return None
        path, leaf = select_result
        if leaf.is_dead_end():
            logging.debug(f"Selection encountered dead end")
            return None
        logging.debug(f"Selection end {path} {leaf}")

        # Expand
        if leaf.is_final():
            logging.debug(
                f"Selected final node with time {leaf.N}, return immediately. "
            )
            if leaf.N >= 100:
                self.stuck_path = path
                return None
            self._increment_virtual_loss(path, self.leaf_num)
            return path, [(path, leaf) for _ in range(self.leaf_num)]

        if len(leaf.get_unexpanded_children()) == 0:
            if not leaf.is_fully_in_tree():
                leaf.reveal_new_children()
                assert leaf.children_count(include_uninitialize=False, on_tree=True) > 0
            else:
                logging.debug(f"{leaf} is fully expanded and has no children.")
                return None

        logging.debug("Expansion start")
        expand_result = self.expand(leaf)
        if expand_result is None:
            logging.debug("Expansion failed.")
            return None
        next_expand, leaf_expanded = expand_result
        path = path.concat(next_expand)
        assert isinstance(path, TreePath), type(path)
        assert not leaf_expanded.is_dead_end()
        logging.debug(f"Expansion end {path}")

        # Simulate
        leaves = self.simulate(path, leaf_expanded)

        if leaves is not None:
            assert len(leaves) == self.leaf_num, leaves
            leaf_expanded.set_alive()
            self._increment_virtual_loss(path, self.leaf_num)
            return path, leaves
        else:
            return None

    #########################
    #   Functional support
    #########################

    def select(self) -> Optional[Tuple[TreePath, TreeNode]]:
        "Find an unexplored descendent of root"
        path = TreePath([])
        node = self.tree_root
        while True:
            node.flush_T(node.N + node._virtual_loss)
            # node is terminal, unexplored, or a leaf
            if node.is_terminal() or len(node.get_unexpanded_children()) > 0:
                return path, node
            if node.children_count() == 0:
                logging.debug(f"{path} has no children")
            assert len(node.get_unexpanded_children()) == 0
            selected = self._ucd_select(node, path)
            if selected is None:
                return None
            next, node, _ = selected
            path = path.concat(next)

    def expand(self, node: TreeNode) -> Optional[Tuple[PseudoTreeNext, TreeNode]]:
        """
        Expand the leaf one level deeper, by choosing a random unexpanded child (N=0). Return None if failed.
        """
        unexpanded_children = node.get_unexpanded_children()
        if len(unexpanded_children) == 0:
            return None

        # randomly select a child from pool
        next, leaf, _ = random.choice(unexpanded_children)
        return next, leaf

    def simulate(
        self,
        tree_path: TreePath,
        leaf_expanded: TreeNode,
        max_final_iterations: int = None,
    ) -> Optional[Tuple[TreePath, TreeNode]]:
        # assert not leaf_expanded.failed_recently
        if leaf_expanded.is_final():
            return [(tree_path, leaf_expanded) for _ in range(self.leaf_num)]

        # leaf_expanded._node.expand(3)

        if max_final_iterations is None:
            a, b, c = self._max_final_iterations
            offset = max(0, (self.simulate_time_ema - self.simulate_decay_time[0]))
            a *= math.exp(-offset / self.simulate_decay_time[1])
            left_depth = self._max_depth - len(tree_path) + 1
            max_final_iterations = int(a * (b**left_depth) + c)

        sample_times = self.leaf_num * max_final_iterations
        assert sample_times > 0
        logging.info(
            f"Getting estimates for path({tree_path}) with {sample_times} samples ..."
        )
        path, dangling_type = tree_path.to_path()
        final_nodes = []
        nodes_set = set()
        for trial_attempt in range(self._sample_retry_times):
            start = time.time()
            trials = self._sampler.random_final_nodes_with_prefix(
                path, sample_times, type=dangling_type, steps=2
            )
            self.simulate_time_ema += time.time() - start
            for trial in trials:
                trial_node = trial.to_node()
                if trial_node not in nodes_set and not (
                    trial_node in self._treenode_store
                    and self._treenode_store[trial_node].filtered
                ):
                    nodes_set.add(trial_node)
                    final_nodes.append(trial)
            if len(final_nodes) >= self.leaf_num:
                break
        final_nodes = list(set([(est.path, est.to_node()) for est in final_nodes]))
        logging.info(
            f"Got {len(final_nodes)} final nodes ({sample_times} samples) for path({tree_path}) after {trial_attempt+1} attempts. "
        )

        if len(final_nodes) < self.leaf_num or leaf_expanded.is_dead_end():
            logging.info(f"Simulation from {tree_path} failed, flushing failure time. ")
            leaf_expanded.set_simulate_fail()
            return None

        self.simulate_time_ema /= 2

        final_nodes = [
            (TreePath(path), self.touch(node, path=path))
            for path, node in random.choices(final_nodes, k=self.leaf_num)
        ]

        return final_nodes

    def back_propagate(
        self, receipt: Receipt, reward: float, path_to_trial: TreePath
    ) -> None:
        """
        Send the reward back up to the ancestors of the leaf
        """

        # check input types
        assert isinstance(reward, float), f"reward={reward} is not a floating number."
        assert isinstance(
            path_to_trial, TreePath
        ), f"path_to_trial={path_to_trial} is not a TreePath."
        # assert 0.0 <= reward <= 1.0
        logging.debug("Back propagation start")
        path = receipt
        self._resurrect(path)
        self._decrement_virtual_loss(path)
        logging.debug("Virtual loss decremented. ")

        # This SHOULD NOT happen when pruning is good enough, but now we are compensating the correctness for efficiency.
        if self.visit(path) is None:
            logging.debug(f'Back propagating a "dead path" {path}...')
            return

        # update rewards
        self.update_nodes(receipt, reward)

        trial_node = self.visit(path_to_trial, on_tree=False)
        assert trial_node and trial_node.is_final()
        try:
            arcs = trial_node.to_node().get_composing_arcs()
        except:
            logging.debug(f"get_composing_arcs failed for {path_to_trial}")
            return

        # update rave scores
        update_types = set([next.type for next in path_to_trial])

        self.update_grave(arcs, update_types, reward)
        logging.debug("Grave updated. ")
        self.update_lrave(trial_node, arcs, reward)
        logging.debug("Lrave updated. ")
        logging.debug("Back propagation end")

    def update_nodes(self, receipt: Receipt, reward: float) -> TreeNode:
        path = receipt

        tree_node = self.tree_root
        if self._policy == "update-descent":
            update_list: List[Tuple[TreeNode, Optional[Arc]]] = [(tree_node, None)]
            for tree_next in path:
                arc = tree_node._node.get_arc_from_handle(tree_next)
                tree_node, _ = tree_node.get_child(tree_next.type, on_tree=True)
                assert tree_node is not None
                if tree_next.key == 0:
                    update_list.append((tree_node, None))
                    break
                else:
                    update_list.append((tree_node, arc))  # mid
                tree_node, _ = tree_node.get_child(tree_next.key, on_tree=True)
                assert tree_node is not None
                update_list.append((tree_node, None))

            for tree_node, arc in update_list:
                tree_node.update(reward, arc)

            final_node = tree_node
        else:
            raise NotImplementedError(
                f"Update policy {self._policy} is not implemented."
            )

        return final_node

    def update_grave(
        self, arcs: Set[Arc], update_types: Set[Next.Type], reward: float
    ) -> None:
        for arc in arcs:
            self.g_rave[arc].update(reward)
        for type in update_types:
            self.g_rave[type].update(reward)

    def update_lrave(
        self, final_node: TreeNode, arcs: List[Arc], reward: float
    ) -> None:
        assert isinstance(final_node, TreeNode), f"{final_node} is not TreeNode!"

        lattice_ends = self._root.expand_to(final_node.to_node())
        attempt_record: Dict[TreeNode, Tuple[bool, Multiset[Arc]]] = dict()

        def find_lattice(
            src_node: TreeNode, tgt_node: TreeNode, arc_pool: Multiset
        ) -> bool:
            """
            Attempt to reach tgt_node from src_node with arcs in arc_pool
            """
            if src_node.to_node() == tgt_node.to_node():
                return True

            if src_node not in attempt_record:
                updated: Set[Tuple[TreeNode, Union[Next.Type, Arc]]] = set()
                for arc in list(arc_pool):
                    child = src_node._node.get_child_from_arc(arc)
                    if child is None:
                        continue
                    new_arc_pool = arc_pool.copy()
                    new_arc_pool.remove(arc, 1)
                    nxt = arc.to_next()
                    mid_child = src_node.get_child(
                        nxt.type,
                        auto_initialize=True,
                        include_simulate_failure=False,
                    )
                    if mid_child is None:
                        continue
                    mid_child = mid_child[0]
                    assert src_node._node in self._path_store, src_node._node
                    child_node = self.touch(
                        src_node._node.get_child_from_arc(arc),
                        path=self._path_store[src_node._node].concat(nxt),
                    )
                    if child_node.to_node().is_dead_end(include_simulate_failure=False):
                        continue
                    if find_lattice(child_node, tgt_node, new_arc_pool):
                        updated.add((src_node, nxt.type))
                        updated.add((mid_child, arc))

                for node, nxt in updated:
                    node.update_lrave(reward, nxt)
                attempt_record[src_node] = (len(updated) > 0, arc_pool)

            assert arc_pool == attempt_record[src_node][1]
            return attempt_record[src_node][0]

        composing_arcs_buffer = {
            lattice_end: Multiset(lattice_end.get_composing_arcs())
            for lattice_end in lattice_ends
        }
        for lattice_start, lattice_end in zip(lattice_ends[:-1], lattice_ends[1:]):
            lattice_arcs = (
                composing_arcs_buffer[lattice_end]
                - composing_arcs_buffer[lattice_start]
            )
            start = self.touch(lattice_start)
            end = self.touch(lattice_end)
            attempt = find_lattice(start, end, lattice_arcs)
            assert (
                attempt
            ), f"Failed to go from {self._path_store[lattice_start]} to {self._path_store[lattice_end]} with lattice_arcs={lattice_arcs}"

    def touch(self, node: Node, path: Path = None) -> TreeNode:
        """
        Initialize node if not exist in store
        """
        assert isinstance(node, Node)
        assert path is None or isinstance(path, Path)
        if node not in self._treenode_store:
            self._treenode_store[node] = TreeNode(self, node)
        if node not in self._path_store:
            if path is None:
                path = node.get_possible_path()
            self._path_store[node] = path
        return self._treenode_store[node]

    def remove(self, receipt: Receipt, trial: TreeNode) -> None:
        "Remove the receipt and set this trial to be dead."
        assert isinstance(trial, TreeNode), type(trial)
        assert trial.is_final(), "The removed trial should be a final node!"
        assert trial == self._treenode_store[trial.to_node()]
        logging.debug("Removing start")

        path = receipt
        self._decrement_virtual_loss(path)
        trial.set_dead()
        trial.filtered = True

        logging.debug("Removing end")

    def _ucd_select(
        self, node: TreeNode, path: TreePath
    ) -> Optional[Tuple[PseudoTreeNext, TreeNode, AverageMeter]]:
        """
        Select a child of node, balancing exploration & exploitation
        """
        children = node.get_children(on_tree=True, filter_simulate_failure=True)

        if len(children) == 0:
            if node.is_fully_in_tree():
                node.set_dead()
            else:
                assert node.reveal_new_children()
            logging.debug("Selection failed. ")
            return None

        # All children of node should already be expanded
        assert len(node.get_unexpanded_children()) == 0, node.get_unexpanded_children()

        log_N_vertex = math.log(node.N)

        def ucb1_tuned(key: Tuple[PseudoTreeNext, TreeNode, AverageMeter]) -> float:
            """
            Upper confidence bound.
            UCB1-tuned.
            """
            nxt, child, edge = key
            if edge.N == 0:
                return 1e9 - self.virtual_loss_constant * child._virtual_loss
            # If the edge is chosen multiple times but the resulting std is very small
            if edge.N > 30 and edge.std <= 0.001:
                logging.info(
                    f"At {path}, encountered {nxt} with {edge} and it is suppressed. "
                )
                return -1e9
            return (
                edge.mean
                + self._exploration_weight
                * math.sqrt(
                    (log_N_vertex / edge.N)
                    * min(
                        0.25, edge.std * edge.std + math.sqrt(2 * log_N_vertex / edge.N)
                    )
                )
                - self.virtual_loss_constant * child._virtual_loss
            )

        selected_child = max(children, key=ucb1_tuned)

        return selected_child

    ##########################
    #      I/O support
    ##########################

    def serialize(self) -> Dict:
        """Serialize the tree and return a dict."""
        logging.debug("Serializing ......")

        # self.garbage_collect(keep_tree_node=True)

        node_list = []
        nodes = list(self._treenode_store.keys())

        # dump father"s n, q
        # dump children"s n, q, filtered
        for i, n in enumerate(nodes):
            father = self._treenode_store[n]
            underlying_node = father._node

            node_serial = {}
            node_serial["index"] = i
            node_serial["node_verbose"] = underlying_node.__repr__()
            assert underlying_node in self._path_store, underlying_node
            # assert underlying_node.is_dead_end() or self._sampler.visit(self._path_store[underlying_node]).to_node() == underlying_node
            node_serial["path"] = self._path_store[underlying_node].serialize()
            node_serial["father"] = {}
            node_serial["father"]["state"] = father.state_dict
            node_serial["father"]["virtual_loss"] = father._virtual_loss
            if father.is_final():
                node_serial["father"]["filtered"] = father.filtered
                node_serial["father"]["reward"] = father.reward
            node_serial["children"] = {}
            for next, mid_child, _ in father.get_children():
                assert isinstance(next, Next.Type)
                next_serial = self.next_serializer.serialize_type(next)
                children_states: Dict[PseudoTreeNext, Tuple[List, Dict[str, Dict]]] = {}
                for n, _, e in mid_child.get_children():
                    arc = underlying_node.get_arc_from_handle(Next(next, n))
                    assert arc is not None
                    # print(f"{mid_child}: {arc} -> {self.g_rave[arc]}")
                    rave_score = {
                        "lrave": mid_child.l_rave[arc].serialize(),
                        "grave": self.g_rave[arc].serialize(),
                    }
                    children_states[str(n)] = e.serialize(), rave_score
                node_serial["children"][next_serial] = {
                    "state": mid_child.state_dict,
                    "virtual_loss": mid_child._virtual_loss,
                    "children": children_states,
                }
            # rave score
            node_serial["lrave"] = {
                self.next_serializer.serialize_type(ty): am.serialize()
                for ty, am in father.l_rave.items()
            }

            node_list.append(node_serial)

        grave_type = {
            self.next_serializer.serialize_type(ty): self.g_rave[ty].serialize()
            for ty in self.g_rave.keys()
            if isinstance(ty, Next.Type)
        }

        packed_args = dict(
            virtual_loss_constant=self.virtual_loss_constant,
            leaf_num=self.leaf_num,
            exploration_weight=self._exploration_weight,
            b=self._b,
            c_l=self._c_l,
            policy=self._policy,
        )
        j = dict(node_list=node_list, args=packed_args, grave_type=grave_type)
        # self.garbage_collect(keep_tree_node=True)
        logging.debug("Serialized.")
        return j

    def serialize_par(self) -> Dict:
        """Serialize the tree and return a dict."""
        logging.debug("Serializing ......")

        # self.garbage_collect(keep_tree_node=True)

        nodes = list(self._treenode_store.keys())
        num_nodes = len(nodes)
        node_list = [None for _ in range(num_nodes)]
        task_queue = queue.Queue(maxsize=len(nodes))
        for i in range(num_nodes):
            task_queue.put(i)

        def consumer():
            while True:
                try:
                    index = task_queue.get_nowait()
                    node = nodes[index]

                    father = self._treenode_store[node]
                    underlying_node = father._node

                    node_serial = {}
                    node_serial["index"] = index
                    node_serial["node_verbose"] = underlying_node.__repr__()
                    assert underlying_node in self._path_store, underlying_node
                    assert (
                        underlying_node.is_dead_end()
                        or self._sampler.visit(
                            self._path_store[underlying_node]
                        ).to_node()
                        == underlying_node
                    )
                    node_serial["path"] = self._path_store[underlying_node].serialize()
                    node_serial["father"] = {}
                    node_serial["father"]["state"] = father.state_dict
                    node_serial["father"]["virtual_loss"] = father._virtual_loss
                    if father.is_final():
                        node_serial["father"]["filtered"] = father.filtered
                        node_serial["father"]["reward"] = father.reward
                    node_serial["children"] = {}
                    for next, mid_child, _ in father.get_children():
                        assert isinstance(next, Next.Type)
                        next_serial = self.next_serializer.serialize_type(next)
                        children_states: Dict[
                            PseudoTreeNext, Tuple[List, Dict[str, Dict]]
                        ] = {}
                        for n, _, e in mid_child.get_children():
                            arc = underlying_node.get_arc_from_handle(Next(next, n))
                            assert arc is not None
                            rave_score = {
                                "lrave": mid_child.l_rave[arc].serialize(),
                                "grave": self.g_rave[arc].serialize(),
                            }
                            children_states[str(n)] = e.serialize(), rave_score
                        node_serial["children"][next_serial] = {
                            "state": mid_child.state_dict,
                            "virtual_loss": mid_child._virtual_loss,
                            "children": children_states,
                        }
                    # rave score
                    node_serial["lrave"] = {
                        self.next_serializer.serialize_type(ty): am.serialize()
                        for ty, am in father.l_rave.items()
                    }
                    node_list[index] = node_serial
                except queue.Empty:
                    return

        threads = [threading.Thread(target=consumer) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        grave_type = {
            self.next_serializer.serialize_type(ty): self.g_rave[ty].serialize()
            for ty in self.g_rave.keys()
            if isinstance(ty, Next.Type)
        }

        assert all(n is not None for n in node_list)

        packed_args = dict(
            virtual_loss_constant=self.virtual_loss_constant,
            leaf_num=self.leaf_num,
            exploration_weight=self._exploration_weight,
            b=self._b,
            c_l=self._c_l,
            policy=self._policy,
        )
        j = dict(node_list=node_list, args=packed_args, grave_type=grave_type)
        # self.garbage_collect(keep_tree_node=True)
        logging.debug("Serialized.")
        return j

    def get_eval_results(self) -> List:
        """Serialize the tree and return a dict."""
        logging.debug("Serializing ......")

        node_list = []
        nodes = list(self._treenode_store.keys())

        # dump father"s n, q
        # dump children"s n, q, filtered
        for i, n in enumerate(nodes):
            father = self._treenode_store[n]
            if not father.is_final() or father.filtered or father.reward == -1:
                continue

            underlying_node = father._node

            assert underlying_node in self._path_store, underlying_node
            node_serial = {
                "key": str(hash(self._path_store[underlying_node].serialize())),
                "path": self._path_store[underlying_node].serialize(),
                "reward": father.reward,
            }

            node_list.append(node_serial)

        logging.debug("Serialized.")
        return node_list

    @staticmethod
    def deserialize(
        serialized: dict,
        sampler: Sampler,
        keep_virtual_loss: bool = False,
        keep_dead_state: bool = False,
        only_load_final: bool = False,
    ) -> "MCTSTree":
        """Deserialize a serialized tree and return a Tree object"""
        logging.debug("Deserializing ......")

        params = serialized["args"]
        node_list = serialized["node_list"]
        tree = MCTSTree(sampler, **params)

        from tqdm import tqdm

        for node_serial in tqdm(node_list):
            if node_serial["father"]["state"]["_is_dead"]:
                if keep_dead_state:
                    path = Path.deserialize(node_serial["path"])
                    underlying_node = tree._sampler.visit(path)
                    father = tree.touch(underlying_node.to_node(), path=path)
                    father.set_dead()
                continue
            path = Path.deserialize(node_serial["path"])
            underlying_node = tree._sampler.visit(path)
            # assert underlying_node is not None, Path.deserialize(node_serial["path"])
            if underlying_node is None:
                continue
            if only_load_final and not underlying_node.is_final():
                continue
            underlying_node = underlying_node.to_node()
            assert underlying_node.__repr__() == node_serial["node_verbose"]
            father = tree.touch(underlying_node, path=path)

            father.load(node_serial["father"]["state"])
            if keep_virtual_loss:
                father._virtual_loss = node_serial["father"]["virtual_loss"]
            if father.is_final():
                father.filtered = node_serial["father"]["filtered"]
                father.reward = node_serial["father"]["reward"]

            for next_serial in node_serial["children"].keys():
                next = tree.next_serializer.deserialize_type(next_serial)
                mid_child = father.get_child(next, auto_initialize=True)[0]
                mid_child.load(node_serial["children"][next_serial]["state"])
                if keep_virtual_loss:
                    mid_child._virtual_loss = node_serial["children"][next_serial][
                        "virtual_loss"
                    ]
                children_states = node_serial["children"][next_serial]["children"]

                for n_sel, (e_sel, rave_score) in children_states.items():
                    n = int(n_sel)
                    e = mid_child.get_child(n, auto_initialize=True)[1]
                    e.refresh(e_sel)
                    arc = underlying_node.get_arc_from_handle(Next(next, n))
                    assert arc is not None
                    mid_child.l_rave[arc].refresh(rave_score["lrave"])
                    tree.g_rave[arc].refresh(rave_score["grave"])
                    # print(f"{arc} -> {tree.g_rave[arc]}")

            # rave score
            for ty_sel, am_sel in node_serial["lrave"].items():
                father.l_rave[tree.next_serializer.deserialize_type(ty_sel)].refresh(
                    am_sel
                )

        for ty_sel, am_sel in serialized["grave_type"].items():
            tree.g_rave[tree.next_serializer.deserialize_type(ty_sel)].refresh(am_sel)

        # tree.garbage_collect(keep_tree_node=True)
        logging.debug("deserialized.")

        return tree

    def garbage_collect(self, keep_tree_node: bool = True):
        """
        Remove non-root tree node with no predecessor.
        All nodes that are accessible from the root should be reserved!
        """
        # Label every alive node
        alive_nodes: Set[Node] = set()

        def label_alive(node: TreeNode) -> bool:
            """Return a boolean value indicating whether the node contains a final descendant."""
            assert isinstance(node, TreeNode)
            children = node.get_children()
            # assert all(not c.is_dead_end() for _, c, _ in children)
            alive_flag = not node.empty() or (keep_tree_node and node._isin_tree)
            for _, child, _ in children:
                child_alive_flag = label_alive(child)
                alive_flag = alive_flag or child_alive_flag
            if alive_flag:
                alive_nodes.add(node._node)
            return alive_flag

        assert label_alive(self.tree_root)

        # Remove dead nodes
        for node, tree_node in list(self._treenode_store.items()):
            if (
                node not in alive_nodes
                and not tree_node.is_final()
                and tree_node.empty()
            ):
                self._treenode_store.pop(node)

        # Clean RAVE dict
        for k, rave_score in list(self.g_rave.items()):
            if rave_score.empty():
                self.g_rave.pop(k)

        for node, tree_node in list(self._treenode_store.items()):
            tree_node.clear_lrave()
            tree_node._virtual_loss = 0
            for child in tree_node.children:
                child.clear_lrave()
                child.clear_edge()
                child._virtual_loss = 0
