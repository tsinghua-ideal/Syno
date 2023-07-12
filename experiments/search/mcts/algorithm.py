import time
import math
import logging
from KAS.Node import Path, Node

from .tree import MCTSTree


class MCTSAlgorithm:
    # TODO: may move to program arguments
    virtual_loss_constant = .5
    leaf_parallelization_number = 1
    exploration_weight = 4 * math.sqrt(2)
    max_iterations = 3000
    b = 0.4
    c_l = 40.

    def __init__(self, sampler, args):
        self.mcts = MCTSTree(sampler, self.virtual_loss_constant, self.leaf_parallelization_number,  self.exploration_weight, self.b, self.c_l)
        self.path_to_meta_data = dict()

    def serialize(self):
        return self.mcts.serialize()

    def update(self, path: Path, reward):
        serialized_path = path.serialize()
        leaf_tree_paths, tree_node, tree_path = self.path_to_meta_data[serialized_path]

        logging.info(f'Updating path: {serialized_path}, reward: {reward}')
        tree_node.reward = reward

        # Back propagate
        if reward < 0:
            for leaf_tree_path in leaf_tree_paths:
                self.mcts.remove(receipt=leaf_tree_path, trial=tree_node)
        else:
            for leaf_tree_path in leaf_tree_paths:
                self.mcts.back_propagate(receipt=leaf_tree_path, reward=reward, path_to_trial=tree_path)

    def launch_new_iteration(self):
        logging.info('Launching new iteration ...')
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
                results[Path(trial_path).serialize()] = (leaf_tree_path, trial_node, trial_path)
            else:
                # Already evaluated, back propagate
                assert not trial_node.filtered
                self.mcts.back_propagate(receipt=leaf_tree_path, reward=trial_node.reward, path_to_trial=trial_path)

        logging.info(f'Iteration finished in {time.time() - start_time} seconds')
        return results
    
    def sample(self):
        n_iterations = 0
        results = []

        while len(results) == 0:
            n_iterations += 1
            iteration_results = self.launch_new_iteration()
            if n_iterations > self.max_iterations:
                return 'retry'
            elif iteration_results is None:
                return 'end'
            
            for path, (leaf_tree_path, trial_node, trial_path) in iteration_results.items():
                if path not in self.path_to_meta_data:
                    self.path_to_meta_data[path] = ([], trial_node, trial_path)
                self.path_to_meta_data[path][0].append(leaf_tree_path)
                results.append(path)
        
        return results
