import time
import math
import logging
from typing import Dict, List, Tuple, Optional
from KAS import Path, Node

from .tree import MCTSTree
from .node import TreePath, TreeNode
from base.models import ManualImpl

class MCTSAlgorithm:
    # TODO: final node has non-consistent virtual loss
    virtual_loss_constant = 0.3
    leaf_parallelization_number = 3
    exploration_weight = math.sqrt(2)
    max_iterations = 3000
    max_final_iterations = 1000
    b = 0.3
    c_l = 10.0
    simulate_retry_period = 8e6
    flush_virtual_loss_period = 300  # Periodically reset virtual loss to 0 (a hack for virtual loss inconsistency) 0 means no flush

    # initial kernels, see base/models/manual_kernels.py for a complete list
    init_kernels = [
        "Conv2d_group_oas",
    ]

    def __init__(self, sampler, args):
        self.mcts = MCTSTree(
            sampler,
            self.virtual_loss_constant,
            self.leaf_parallelization_number,
            self.exploration_weight,
            self.b,
            self.c_l,
            max_final_iterations=self.max_final_iterations,
            simulate_retry_period=self.simulate_retry_period
        )
        self.sampler = sampler
        self.path_to_meta_data = dict()

        self.sample_num = 0
        self.time_stamp = time.time()

        self.preconditioned = "gpt" in args.model

    def serialize(self):
        return self.mcts.serialize()

    def deserialize(self, serialized_dict):
        self.preconditioned = True
        self.mcts.deserialize(serialized_dict, self.sampler, keep_dead_state=False)

    def update(self, path: Path, reward):
        serialized_path = path.serialize()
        leaf_tree_paths, tree_node, tree_path = self.path_to_meta_data[serialized_path]

        logging.info(f"Updating path: {serialized_path}, reward: {reward}")
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

    def launch_new_iteration(self) -> Dict[str, Tuple[TreePath, TreeNode, TreePath]]:
        logging.info("Launching new iteration ...")
        start_time = time.time()

        rollout = self.mcts.do_rollout()
        if rollout is None:
            return None

        results = dict()
        assert len(rollout) > 0
        leaf_tree_path, trials = rollout
        for trial_path, trial_node in trials:
            assert trial_node.is_final()
            if trial_node.reward < 0:
                # Unevaluated
                results[Path(trial_path).serialize()] = (
                    leaf_tree_path,
                    trial_node,
                    trial_path,
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
            for k in list(self.mcts.virtual_loss_count.keys()):
                self.mcts.virtual_loss_count[k] = 0
            logging.debug("Virtual losses cleared. ")
            
        n_iterations = 0
        results = []
        
        if not self.preconditioned:
            impl = ManualImpl(self.sampler)
            for kernel_name in self.init_kernels:
                assert hasattr(impl, kernel_name), f"{kernel_name} is not a valid kernel"
                kernel = getattr(impl, kernel_name)()
                trial_path = Path(kernel.convert_to_path(self.sampler))
                logging.info(f"This MCTS is pre-conditioned on kernel {kernel_name} with path {trial_path}...")
                trial_node = self.mcts.visit(trial_path, on_tree=False, put_in_tree=True)
                path = Path(trial_path).serialize()
                assert trial_node is not None, f"Kernel {kernel_name} is outside the search space!"
                assert path not in self.path_to_meta_data
                self.path_to_meta_data[path] = (list(trial_path.hierarchy), trial_node, trial_path)
                results.append(path)
            self.preconditioned = True

        while len(results) == 0:
            n_iterations += 1
            iteration_results = self.launch_new_iteration()
            if n_iterations > self.max_iterations:
                return "retry"
            elif iteration_results is None:
                return "end"

            for path, (
                leaf_tree_path,
                trial_node,
                trial_path,
            ) in iteration_results.items():
                if path not in self.path_to_meta_data:
                    self.path_to_meta_data[path] = ([], trial_node, trial_path)
                    results.append(path)
                self.path_to_meta_data[path][0].append(leaf_tree_path)

        self.sample_num += 1

        return results
