import time
import math
import logging
from typing import Dict, List, Tuple, Optional
from KAS.Node import Path, Node

from .tree import MCTSTree
from .node import TreePath, TreeNode


class MCTSAlgorithm:
    # TODO: final node has non-consistent virtual loss
    virtual_loss_constant = 0.3
    leaf_parallelization_number = 3
    exploration_weight = 4 * math.sqrt(2)
    max_iterations = 3000
    max_final_iterations = 2000
    b = 0.5
    c_l = 20.0
    simulate_retry_period = 8e5
    flush_virtual_loss_period = 0  # Periodically reset virtual loss to 0 (a hack for virtual loss inconsistency) 0 means no flush

    init_paths = [
        "[Reduce(15279624404278704057), Reduce(15279624404278704057), Reduce(5554949874972930775), Share(13407255124661549680), Unfold(1084665757264077127), Share(9989589238219652823), Share(7115518979942117690), Unfold(15903218782717349037), Finalize(6565149261286651715)]",  # Conv2d
        # "[Reduce(5554949874972930775), Share(9989589238219652823), Finalize(13643907174028270664)]",  # Linear
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

        self.init_samples: List[Tuple[TreePath, TreeNode]] = []
        for path_sel in self.init_paths:
            path = TreePath.decode_str(path_sel)
            trial_node = self.mcts.visit(path, on_tree=False, put_in_tree=True)
            assert trial_node is not None
            self.init_samples.append((path, trial_node))

    def serialize(self):
        return self.mcts.serialize()

    def deserialize(self, serialized_dict):
        self.mcts.deserialize(serialized_dict, self.sampler, keep_dead_state=True)

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
        if self.init_samples is not None:
            logging.info("Injecting bootstrapping path ...")
            results = dict()
            for (
                trial_path,
                trial_node,
            ) in self.init_samples:
                for path in trial_path.hierarchy:
                    results[Path(path).serialize()] = (
                        path,
                        trial_node,
                        trial_path,
                    )
            self.init_samples = None
            return results

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
        n_iterations = 0
        results = []

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
        if (
            self.flush_virtual_loss_period > 0
            and self.sample_num % self.flush_virtual_loss_period == 0
        ):
            logging.debug("Resetting virtual losses... ")
            for k in list(self.mcts.virtual_loss_count.keys()):
                self.mcts.virtual_loss_count[k] = 0
            logging.debug("Virtual losses cleared. ")

        return results
