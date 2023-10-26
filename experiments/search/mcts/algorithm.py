import time
import math
import logging
from typing import Dict, List, Tuple, Optional
from KAS import Path, Node

from .tree import MCTSTree
from .node import TreePath, TreeNode
from .MCTSExplorer import MCTSExplorer
from base.models import ManualImpl


class MCTSAlgorithm:
    virtual_loss_constant = 0.3
    leaf_parallelization_number = 3
    exploration_weight = 4 * math.sqrt(2)
    max_iterations = 3000
    max_final_iterations = (6, 1.5, 1000)
    b = 0.8
    c_l = 10.0
    simulate_retry_period = 1e9
    sample_retry_times = 5
    rave_random_ratio = 0.3
    simulate_decay_time = (30, 120)
    flush_virtual_loss_period = 2400  # Periodically reset virtual loss to 0 (a hack for virtual loss inconsistency) 0 means no flush

    # initial kernels, see base/models/manual_kernels.py for a complete list
    init_kernels = [
        # "Conv2d_group_oas",
    ]

    def __init__(self, sampler, model, args):
        self.mcts = MCTSTree(
            sampler,
            self.virtual_loss_constant,
            self.leaf_parallelization_number,
            self.exploration_weight,
            self.b,
            self.c_l,
            max_final_iterations=self.max_final_iterations,
            simulate_retry_period=self.simulate_retry_period,
            sample_retry_times=self.sample_retry_times,
            simulate_decay_time=self.simulate_decay_time, 
            max_depth=args.kas_depth,
            rave_random_ratio=self.rave_random_ratio
        )
        self.sampler = sampler
        self.explorer = MCTSExplorer(model, sampler, self.mcts)
        self.path_toupd: Dict[
            Node, Tuple[TreePath, List[TreePath]]
        ] = dict()  # trial node to meta data

        self.sample_num = 0
        self.time_stamp = time.time()

        self.preconditioned = "gpt" in args.model

    def serialize(self):
        return self.mcts.serialize()

    def deserialize(self, serialized_dict):
        # self.preconditioned = True
        self.mcts.deserialize(serialized_dict, self.sampler, keep_dead_state=False)
        
    def dump_eval_result(self):
        return self.mcts.get_eval_results()
    
    def load_eval_result(self, path_serialized, reward, leaf_path=None):
        path = Path.deserialize(path_serialized)
        node = self.mcts.visit(
            path, on_tree=False, put_in_tree=True
        )
        if node is None:
            return
        if leaf_path is None:
            for leaf_path in path.hierarchy:
                n = self.mcts.visit(leaf_path, on_tree=False, put_in_tree=True)
                if n.N == 0:
                    break
            
        self.path_toupd[node.to_node()] = (TreePath(path), [leaf_path])
        self.mcts._increment_virtual_loss(leaf_path, 1)
        self.update(path, reward)

    def update(self, path: Path, reward):
        tree_node = self.mcts.visit(TreePath(path), on_tree=False)
        if tree_node is None:
            logging.warning(f"{path} is not in our space. Skipping")
            return

        tree_path, leaf_tree_paths = self.path_toupd.pop(tree_node.to_node())

        logging.info(f"Updating path: {leaf_tree_paths} ({tree_path}), reward: {reward}")
        tree_node.reward = reward

        # Back propagate
        if reward < 0:
            for leaf_tree_path in leaf_tree_paths:
                self.mcts.remove(receipt=leaf_tree_path, trial=tree_node)
        else:
            for leaf_tree_path in leaf_tree_paths:
                self.mcts.back_propagate(
                    receipt=leaf_tree_path, reward=reward, path_to_trial=tree_path
                )

    def launch_new_iteration(self) -> List[Tuple[TreePath, TreeNode, TreePath]]:
        logging.info("Launching new iteration ...")
        start_time = time.time()

        rollout = self.mcts.do_rollout()
        if rollout is None:
            return None

        results = []
        assert len(rollout) > 0
        leaf_tree_path, trials = rollout
        for trial_path, trial_node in trials:
            assert trial_node.is_final()
            if trial_node.reward < 0:
                # Unevaluated
                results.append(
                    (
                        leaf_tree_path,
                        trial_node,
                        trial_path,
                    )
                )
            else:
                # Already evaluated, back propagate
                assert not trial_node.filtered
                self.mcts.back_propagate(
                    receipt=leaf_tree_path,
                    reward=trial_node.reward,
                    path_to_trial=trial_path,
                )

        logging.info(f"Iteration finished in {time.time() - start_time} seconds")
        return results

    def sample(self):

        if (
            self.flush_virtual_loss_period > 0
            and time.time() - self.time_stamp > self.flush_virtual_loss_period
        ):
            self.time_stamp = time.time()
            logging.debug("Resetting virtual losses... ")
            for v in self.mcts._treenode_store.values():
                v._virtual_loss = 0
            logging.debug("Virtual losses cleared. ")

        n_iterations = 0
        results = []

        if not self.preconditioned:
            impl = ManualImpl(self.sampler)
            for kernel_name in self.init_kernels:
                assert hasattr(
                    impl, kernel_name
                ), f"{kernel_name} is not a valid kernel"
                kernel = getattr(impl, kernel_name)()
                trial_path = TreePath(kernel.convert_to_path(self.sampler))
                logging.info(
                    f"This MCTS is pre-conditioned on kernel {kernel_name} with path {trial_path}..."
                )
                trial_node = self.mcts.visit(
                    trial_path, on_tree=False, put_in_tree=True
                )
                path = Path(trial_path).serialize()
                assert (
                    trial_node is not None
                ), f"Kernel {kernel_name} is outside the search space!"
                assert trial_node.to_node() not in self.path_toupd
                hierarchy = list(trial_path.hierarchy)
                self.path_toupd[trial_node.to_node()] = (trial_path, hierarchy)
                for path_to_hierarchy in hierarchy:
                    self.mcts._increment_virtual_loss(path_to_hierarchy, 1)
                results.append(path)
            self.preconditioned = True

        while len(results) == 0:
            n_iterations += 1
            iteration_results = self.launch_new_iteration()
            if n_iterations > self.max_iterations:
                return "retry"
            elif iteration_results is None:
                return "end"

            for leaf_tree_path, trial_node, trial_path in iteration_results:
                if trial_node.to_node() not in self.path_toupd:
                    self.path_toupd[trial_node.to_node()] = (trial_path, [])
                    results.append(Path(trial_path).serialize())
                self.path_toupd[trial_node.to_node()][1].append(leaf_tree_path)

        self.sample_num += 1

        return results
