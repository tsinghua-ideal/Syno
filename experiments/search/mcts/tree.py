import random
import math
import logging
import time
from collections import defaultdict
from typing import List, Tuple, Optional, DefaultDict, Set, Union, Dict
from tqdm import tqdm

# Multi-thread support
import threading
import queue

from KAS.Node import Node, Path
from KAS.Sampler import Sampler
from KAS.Bindings import Next, Arc
from KAS.Utils import NextSerializer

from .node import TreeNode, TreePath, PseudoTreeNext, PseudoArc
from .avg_meter import AverageMeter


MockArc = Union[Next, Arc]


def join_all(threads, timeout) -> bool:
    """
    Reference: https://stackoverflow.com/questions/24065808/python-join-multiple-threads-with-timeout
    Args:
        threads: a list of thread objects to join
        timeout: the maximum time to wait for the threads to finish
    Return:
        Whether all threads are finished.
    """
    start = cur_time = time.time()
    while cur_time <= (start + timeout):
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=0)
        if all(not t.is_alive() for t in threads):
            return True
        time.sleep(0.1)
        cur_time = time.time()
    else:
        still_running = [t for t in threads if t.is_alive()]
        num = len(still_running)
        return num == 0


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
        time_limits: List[Tuple[int, bool]] = [(30, False)],
        kas_mcts_workers: int = 4,
    ) -> None:
        self._treenode_store: Dict[Node, TreeNode] = dict()
        self._root = sampler.root().to_node()
        assert not self._root.is_dead_end()
        self.tree_root = TreeNode(self, self._root)
        self._treenode_store[self._root] = self.tree_root
        self.g_rave: DefaultDict[PseudoArc, AverageMeter] = defaultdict(AverageMeter)

        self._sampler = sampler
        self._exploration_weight = exploration_weight
        self._c_l = c_l
        self._b = b
        self._policy = policy
        self._time_limits = time_limits
        self._kas_mcts_workers = kas_mcts_workers
        assert policy in ["update-all", "update-descent"]
        random.seed(sampler._seed)

        # Tree Parallelization
        self.virtual_loss_constant = virtual_loss_constant
        self.virtual_loss_count: DefaultDict[TreeNode, int] = defaultdict(
            int
        )  # node -> virtual loss count

        # Leaf Parallelization
        self.leaf_num = leaf_num

        self.next_serializer = NextSerializer()

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
        logging.debug(f"increment virtual loss by {delta} of {path}")
        tree_node = self.tree_root
        self.virtual_loss_count[tree_node] += delta
        logging.debug(
            f"increment virtual loss by {delta} at {tree_node}, virtual loss count: {self.virtual_loss_count[tree_node] - delta} -> {self.virtual_loss_count[tree_node]}"
        )
        for next in path:
            tree_node = tree_node.get_child(next.type, on_tree=True)
            if tree_node is None:
                logging.debug(
                    "Error: TreeNode is none, maybe due to another prefetcher"
                )
                logging.debug(f"stopped at {next.type}")
                break
            tree_node, _ = tree_node
            self.virtual_loss_count[tree_node] += delta
            logging.debug(
                f"increment virtual loss by {delta} at {tree_node}, virtual loss count: {self.virtual_loss_count[tree_node] - delta} -> {self.virtual_loss_count[tree_node]}"
            )
            if next.key == 0:
                break
            tree_node = tree_node.get_child(next.key, on_tree=True)
            if tree_node is None:
                logging.debug(
                    "Error: TreeNode is none, maybe due to another prefetcher"
                )
                logging.debug(f"stopped at {next.key}")
                break
            tree_node, _ = tree_node
            self.virtual_loss_count[tree_node] += delta
            logging.debug(
                f"increment virtual loss by {delta} at {tree_node}, virtual loss count: {self.virtual_loss_count[tree_node] - delta} -> {self.virtual_loss_count[tree_node]}"
            )

    def _decrement_virtual_loss(self, path: TreePath, delta: int = 1) -> None:
        assert delta > 0
        logging.debug(f"decrement virtual loss by {delta} of {path}")
        tree_node = self.tree_root
        self.virtual_loss_count[tree_node] -= delta
        logging.debug(
            f"decrement virtual loss by {delta} at {tree_node}, virtual loss count: {self.virtual_loss_count[tree_node] + delta} -> {self.virtual_loss_count[tree_node]}"
        )
        if self.virtual_loss_count[tree_node] < 0:
            self.virtual_loss_count[tree_node] = 0
            logging.warn("Error: Virtual loss go below 0! ")
        for next in path:
            tree_node = tree_node.get_child(next.type, on_tree=True)
            # assert tree_node is not None
            if tree_node is None:
                logging.debug(f"stopped at {next.type}")
                break
            tree_node, _ = tree_node
            self.virtual_loss_count[tree_node] -= delta
            logging.debug(
                f"decrement virtual loss by {delta} at {tree_node}, virtual loss count: {self.virtual_loss_count[tree_node] + delta} -> {self.virtual_loss_count[tree_node]}"
            )
            if self.virtual_loss_count[tree_node] < 0:
                self.virtual_loss_count[tree_node] = 0
                logging.warn("Error: Virtual loss go below 0! ")
            if next.key == 0:
                break
            tree_node = tree_node.get_child(next.key, on_tree=True)
            # assert tree_node is not None
            if tree_node is None:
                logging.debug(f"stopped at {next.key}")
                break
            tree_node, _ = tree_node
            self.virtual_loss_count[tree_node] -= delta
            logging.debug(
                f"decrement virtual loss by {delta} at {tree_node}, virtual loss count: {self.virtual_loss_count[tree_node] + delta} -> {self.virtual_loss_count[tree_node]}"
            )
            if self.virtual_loss_count[tree_node] < 0:
                self.virtual_loss_count[tree_node] = 0
                logging.warn("Error: Virtual loss go below 0! ")

    def visit(self, path: TreePath, on_tree: bool = True) -> Optional[TreeNode]:
        tree_node = self.tree_root
        for next in path:
            tree_node = tree_node.get_child(
                next.type, auto_initialize=not on_tree, on_tree=on_tree
            )
            if tree_node is None:
                return None
            tree_node, _ = tree_node
            if next.key == 0:
                break
            tree_node = tree_node.get_child(
                next.key, auto_initialize=not on_tree, on_tree=on_tree
            )
            if tree_node is None:
                return None
            tree_node, _ = tree_node
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
        select_result = self._select()
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
        expand_result = self._expand(leaf)
        if expand_result is None:
            logging.debug("Expansion failed.")
            return None
        next_expand, leaf_expanded = expand_result
        path = path.concat(next_expand)
        assert isinstance(path, TreePath), type(path)
        assert not leaf_expanded.is_dead_end()
        self._increment_virtual_loss(path, self.leaf_num)
        logging.debug(f"Expansion end {path}")

        # Simulate
        leaves_queue = queue.Queue(maxsize=self.leaf_num)
        logging.debug(f"Simulation start")
        leaf_expanded._node.expand(3)
        logging.debug(f"Expanded 3 layers")

        # Multi-thread object
        def thread_simulate(path: TreePath, node: TreeNode):
            # logging.debug(f"Thread {threading.get_ident()} start")
            while not leaves_queue.full() and not node.is_dead_end():
                leaf_simul = self._simulate(path, node)
                if leaf_simul is not None:
                    try:
                        leaves_queue.put_nowait(leaf_simul)
                    except queue.Full:
                        assert leaves_queue.full()
                        pass
            # logging.debug(f"Thread {threading.get_ident()} end (full={leaves_queue.full()}, dead={node.is_dead_end()})")

        threads = [
            threading.Thread(target=thread_simulate, args=(path, leaf_expanded))
            for _ in range(self._kas_mcts_workers)
        ]
        for thread in threads:
            thread.start()

        last_limit = 0
        failure_flag = False
        for i, (time_limit, blocking) in enumerate(self._time_limits):
            if not join_all(threads, timeout=time_limit - last_limit):
                logging.debug(f"Timeout for {i+1} times. ")
                if i < len(self._time_limits) - 1:
                    if blocking:
                        leaf_expanded._node.expand(4 + i)
                    else:
                        leaf_expanded._node.expand_async(4 + i)
                elif leaf_expanded.is_alive():
                    # There are successful attempts before. We cannot set it to dead now
                    failure_flag = True
                    leaf_expanded._node.expand_async(4 + i)
                    logging.debug(
                        f"Simulation from {path} failed, not setting dead because of previous successful attempts. "
                    )
                else:
                    failure_flag = True
                    logging.debug(
                        f"Force {path} to be dead because simulate failed for too many times. "
                    )
                    leaf_expanded.set_dead()
                    assert leaf_expanded.is_dead_end()
                last_limit = time_limit
            else:
                break

        if leaf_expanded.is_dead_end():
            failure_flag = True

        leaves: List[TreeNode] = list(leaves_queue.queue)

        if failure_flag:
            self._decrement_virtual_loss(path, self.leaf_num)
            return None
        else:
            assert len(leaves) == self.leaf_num, leaves
            leaf_expanded.set_alive()
            assert leaf_expanded.is_alive()
            return path, leaves

    #########################
    #   Functional support
    #########################

    def _select(self) -> Optional[Tuple[TreePath, TreeNode]]:
        "Find an unexplored descendent of root"
        path = TreePath([])
        node = self.tree_root
        while True:
            node.flush_T(node.N + self.virtual_loss_count[node])
            # node is terminal, unexplored, or a leaf
            if node.is_terminal() or len(node.get_unexpanded_children()) > 0:
                return path, node
            assert len(node.get_unexpanded_children()) == 0
            selected = self._ucd_select(node)
            if selected is None:
                return None
            next, node, _ = selected
            path = path.concat(next)

    def _expand(self, node: TreeNode) -> Optional[Tuple[PseudoTreeNext, TreeNode]]:
        """
        Expand the leaf one level deeper, by choosing a random unexpanded child (N=0). Return None if failed.
        """
        unexpanded_children = node.get_unexpanded_children()
        if len(unexpanded_children) == 0:
            return None

        # randomly select a child from pool
        next, leaf, _ = random.choice(unexpanded_children)
        return next, leaf

    def _simulate(
        self, path: TreePath, node: TreeNode
    ) -> Optional[Tuple[TreePath, TreeNode]]:
        "Returns a random simulation (to completion) of `node`"

        def random_child(
            node: TreeNode,
        ) -> Optional[Tuple[TreePath, TreeNode, AverageMeter]]:
            """
            Two step random selection. First, randomly select a primitive type. Then, randomly select a child of that type.
            """
            assert isinstance(node, TreeNode)
            nexts = node.get_children_nexts()
            if len(nexts) == 0:
                # logging.debug(f"Simulation Encountered dead father {node}")
                assert node.is_dead_end()
                return None
            next = random.choice(nexts)
            child = node.get_child(next, auto_initialize=True)
            if child:
                selected_child, edge_state = child
                return next, selected_child, edge_state
            else:
                # logging.debug(f"Simulation Encountered dead child {node} -> {next}")
                return None

        while not node.is_terminal():
            random_select_result = random_child(node)
            if random_select_result is None:
                return None
            next, node, edge_state = random_select_result
            path = path.concat(next)

        if node.is_final():
            return path, node
        else:
            # logging.debug(f"Simulation Encountered dead end {node}")
            return None

    def back_propagate(
        self, receipt: Receipt, reward: float, path_to_trial: TreePath
    ) -> None:
        """
        Send the reward back up to the ancestors of the leaf
        """

        # check input types
        assert isinstance(reward, float)
        assert isinstance(path_to_trial, TreePath)
        # assert 0.0 <= reward <= 1.0
        logging.debug("Back propagation start")
        path = receipt
        self._decrement_virtual_loss(path)

        # This SHOULD NOT happen when pruning is good enough, but now we are compensating the correctness for efficiency.
        if self.visit(path) is None:
            logging.debug(f'Back propagating a "dead path" {path}...')
            return

        # update rewards
        self.update_nodes(receipt, reward)

        trial_node = self._sampler.visit(path_to_trial)
        assert trial_node.is_final()
        try:
            arcs = trial_node.get_composing_arcs()
        except:
            logging.debug(path_to_trial)
            return

        # update rave scores
        update_types = set([next.type for next in path_to_trial])
        trial_node_on_tree = self.touch(trial_node.to_node())

        self.update_grave(arcs, update_types, reward)
        self.update_lrave(trial_node_on_tree, arcs, reward)
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
        elif self._policy == "update-all":
            logging.warning(
                "Update-all policy is still not fully implemented. Please do not use it for now!"
            )
            for tree_next in path:
                arc = tree_node._node.get_arc_from_handle(tree_next)
                tree_node, _ = tree_node.get_child(tree_next.type, on_tree=True)
                if tree_next.key == 0:
                    break
                tree_node, _ = tree_node.get_child(tree_next.key, on_tree=True)
            assert tree_node is not None
            final_node = tree_node
            logging.debug(f"final node is {final_node}")
            arcs = final_node._node.get_composing_arcs()

            # Mark Ancestors
            ancestors: List[TreeNode] = []

            def mark_ancestors(
                src_node: TreeNode, tgt_node: TreeNode, arc_pool: Set[Arc]
            ) -> bool:
                logging.debug(f"mark_ancestors({src_node}, {tgt_node}, {arc_pool})")
                if src_node == tgt_node or tgt_node in src_node.children:
                    return True

                updated_flag = False
                for arc in arc_pool:
                    if src_node._node.can_accept_arc(arc):
                        arc_pool.remove(arc)
                        nxt = arc.to_next()
                        mid_child = src_node.get_child(nxt.type, auto_initialize=True)[
                            0
                        ]
                        assert mid_child is not None
                        child_node = self.touch(src_node._node.get_child_from_arc(arc))
                        if (
                            mark_ancestors(child_node, tgt_node, arc_pool)
                            and not updated_flag
                        ):
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

            def compute_weights(
                src_node: TreeNode,
                tgt_node: TreeNode,
                arc_pool: Set[Arc],
                current_weight: float,
                alpha: float,
            ) -> bool:
                logging.debug(
                    f"compute_weights({src_node}, {tgt_node}, {arc_pool}, {current_weight})"
                )
                if src_node == tgt_node or tgt_node in src_node.children:
                    return True

                updated_flag = False
                potential_children: Dict[TreeNode, Set[TreeNode]] = defaultdict(set)

                for arc in arc_pool:
                    if src_node._node.can_accept_arc(arc):
                        arc_pool.remove(arc)
                        nxt = arc.to_next()
                        mid_child = src_node.get_child(nxt.type, auto_initialize=True)[
                            0
                        ]
                        assert mid_child is not None
                        child_node = self.touch(src_node._node.get_child_from_arc(arc))
                        if child_node in ancestors and not updated_flag:
                            potential_children[mid_child].add(child_node)
                            edge_buffer[mid_child].add(nxt.key)
                            updated_flag = True
                        arc_pool.add(arc)

                if updated_flag:
                    mid_child_weight: float = current_weight / len(
                        potential_children.keys()
                    )
                    for child_node, child_nodes in potential_children.items():
                        node_weights[child_node] += mid_child_weight
                        for child_node in child_nodes:
                            child_weight = mid_child_weight / len(child_nodes)
                            node_weights[child_node] += child_weight
                            logging.debug(
                                f"node_weights[{child_node}] += {child_weight}"
                            )

                return updated_flag

            queue = [self.tree_root]
            while len(queue) > 0:
                node = queue.pop(0)
                compute_weights(node, final_node, set(arcs), node_weights[node], 1.0)
            logging.debug(
                f"weights computed. node_weights: {node_weights}, edge_buffer: {edge_buffer}"
            )

            # Update
            for tree_node_toupd, weight in node_weights.items():
                tree_node_toupd.update(reward * weight)
                if tree_node_toupd._is_mid:
                    for key in edge_buffer[tree_node_toupd]:
                        tree_node_toupd.update_edge(
                            reward * weight / len(edge_buffer[tree_node_toupd]), key
                        )

        return final_node

    def update_grave(
        self, arcs: Set[Arc], update_types: Set[Next.Type], reward: float
    ) -> None:
        for arc in arcs:
            self.g_rave[arc].update(reward)
        for type in update_types:
            self.g_rave[type].update(reward)

    def update_lrave(self, final_node: TreeNode, arcs: Set[Arc], reward: float) -> None:
        assert isinstance(final_node, TreeNode), f"{final_node} is not TreeNode!"

        def attempt_to_node(
            src_node: TreeNode, tgt_node: TreeNode, arc_pool: Set[Arc]
        ) -> bool:
            """
            Attempt to reach tgt_node from src_node with arcs in arc_pool
            """
            if src_node == tgt_node:
                return True

            updated: Set[Tuple[TreeNode, Union[Next.Type, Arc]]] = set()

            for arc in list(arc_pool):
                if src_node._node.can_accept_arc(arc):
                    arc_pool.remove(arc)
                    nxt = arc.to_next()
                    mid_child = src_node.get_child(nxt.type, auto_initialize=True)
                    if mid_child is None:
                        arc_pool.add(arc)
                        continue
                    mid_child = mid_child[0]
                    child_node = self.touch(src_node._node.get_child_from_arc(arc))
                    if child_node.is_dead_end():
                        arc_pool.add(arc)
                        continue
                    if attempt_to_node(child_node, tgt_node, arc_pool):
                        updated.add((src_node, nxt.type))
                        updated.add((mid_child, arc))
                    arc_pool.add(arc)

            for node, nxt in updated:
                node.update_lrave(reward, nxt)
            return len(updated) > 0

        attempt_to_node(self.tree_root, final_node, set(arcs))

    def touch(self, node: Node) -> TreeNode:
        """
        Initialize node if not exist in store
        """
        assert isinstance(node, Node)
        if node not in self._treenode_store:
            self._treenode_store[node] = TreeNode(self, node)
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
        self, node: TreeNode
    ) -> Optional[Tuple[PseudoTreeNext, TreeNode, AverageMeter]]:
        """
        Select a child of node, balancing exploration & exploitation
        """
        children = node.get_children(on_tree=True)

        if len(children) == 0:
            if not node.is_fully_in_tree():
                node.reveal_new_children(self.g_rave, self._c_l)
            logging.debug("Selection failed. ")
            return None

        # All children of node should already be expanded
        assert len(node.get_unexpanded_children()) == 0

        log_N_vertex = math.log(node.N)

        def ucb1_tuned(key: Tuple[PseudoTreeNext, TreeNode, AverageMeter]) -> float:
            """
            Upper confidence bound.
            UCB1-tuned.
            """
            _, child, edge = key
            if edge.N == 0:
                return 1e9 - self.virtual_loss_constant * self.virtual_loss_count[child]
            return (
                edge.mean
                + self._exploration_weight
                * math.sqrt(
                    (log_N_vertex / edge.N)
                    * min(
                        0.25, edge.std * edge.std + math.sqrt(2 * log_N_vertex / edge.N)
                    )
                )
                - self.virtual_loss_constant * self.virtual_loss_count[child]
            )

        selected_child = max(children, key=ucb1_tuned)

        return selected_child

    ##########################
    #      I/O support
    ##########################

    def serialize(self) -> Dict:
        """Serialize the tree and return a dict."""
        logging.debug("Serializing ......")

        self.garbage_collect(keep_tree_node=True)

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
            node_serial["path"] = underlying_node.get_possible_path().serialize()
            node_serial["father"] = {}
            node_serial["father"]["state"] = father.state_dict
            node_serial["father"]["virtual_loss"] = self.virtual_loss_count[father]
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
                    "virtual_loss": self.virtual_loss_count[mid_child],
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

    @staticmethod
    def deserialize(
        serialized: dict,
        sampler: Sampler,
        keep_virtual_loss: bool = False,
        keep_dead_state: bool = False,
    ) -> "MCTSTree":
        """Deserialize a serialized tree and return a Tree object"""
        logging.debug("Deserializing ......")

        params = serialized["args"]
        node_list = serialized["node_list"]
        tree = MCTSTree(sampler, **params)

        for node_serial in tqdm(node_list):
            if node_serial["father"]["state"]["_is_dead"]:
                if keep_dead_state:
                    underlying_node = tree._sampler.visit(
                        Path.deserialize(node_serial["path"])
                    )
                    father = tree.touch(underlying_node)
                    father.set_dead()
                continue
            underlying_node = tree._sampler.visit(Path.deserialize(node_serial["path"]))
            # assert underlying_node is not None, Path.deserialize(node_serial["path"])
            if underlying_node is None:
                continue
            underlying_node = underlying_node.to_node()
            assert underlying_node.__repr__() == node_serial["node_verbose"]
            father = tree.touch(underlying_node)

            father.load(node_serial["father"]["state"])
            if keep_virtual_loss:
                tree.virtual_loss_count[father] = node_serial["father"]["virtual_loss"]
            if father.is_final():
                father.filtered = node_serial["father"]["filtered"]
                father.reward = node_serial["father"]["reward"]

            for next_serial in node_serial["children"].keys():
                next = tree.next_serializer.deserialize_type(next_serial)
                mid_child = father.get_child(next, auto_initialize=True)[0]
                mid_child.load(node_serial["children"][next_serial]["state"])
                if keep_virtual_loss:
                    tree.virtual_loss_count[mid_child] = node_serial["children"][
                        next_serial
                    ]["virtual_loss"]
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
            assert all(not c.is_dead_end() for _, c, _ in children)
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
            for child in tree_node.children:
                child.clear_lrave()
                child.clear_edge()

        # Clean virtual loss dict
        for k, v in list(self.virtual_loss_count.items()):
            if v == 0:
                self.virtual_loss_count.pop(k)
