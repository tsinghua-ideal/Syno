import math

# Systems
import os
import json
import logging
from time import time, sleep
from typing import Dict, List, Tuple, Any

# KAS
from KAS import MCTS, Sampler, Statistics, Node, TreeNode


class MCTSTrainer:
    def __init__(self,
                 sampler: Sampler,
                 arguments: Dict,
                 mcts_iterations: int = 1000,
                 leaf_parallelization_number: int = 5,
                 virtual_loss_constant: float = 5.,
                 exploration_weight: float = math.sqrt(2)
                 ) -> None:
        
        self.mcts = MCTS(sampler, virtual_loss_constant,
                         leaf_parallelization_number, exploration_weight)

        print("MCTS initialized")

        # flags
        self.best_node = (0, 0)
        self.reward_list: List[float] = []
        self.time_list: List[float] = []
        self.end_flag: bool = False
        self.start_time: float = time()

        # arguments
        self.args = arguments
        self.remain_iterations = mcts_iterations
        self.leaf_parallelization_number = leaf_parallelization_number
        self.virtual_loss_constant = virtual_loss_constant

    def has_eval_result(self, node: TreeNode) -> bool:
        return node.reward != -1

    def check_dead(self, node: TreeNode) -> bool:
        assert node.is_final()
        return node.filtered

    def get_eval_result(self, node: TreeNode) -> float:
        assert not self.check_dead(node), "Node is dead!"
        assert self.has_eval_result(node), "No reward!"
        return node.reward

    def set_eval_result(self, node: TreeNode, val: float) -> None:
        assert 0 <= val <= 1, "Invalid eval result!"
        assert not self.has_eval_result(node), "Node already has eval result!"
        node.reward = val

    def get_args(self) -> Dict:
        return self.args

    def update_result(self, meta: Tuple[TreeNode, Any], reward: float) -> None:
        """preprocess a path after evaluation. """
        trial, receipt = meta
        if reward == -1:
            self.mcts.remove(receipt, trial)
        else:
            self.set_eval_result(trial, reward)
            # update
            if self.best_node[1] < reward:
                self.best_node = (trial, reward)
            self.mcts.back_propagate(receipt, reward)
            self.reward_list.append(reward)
            self.time_list.append(time() - self.start_time)
            print("Successfully updated MCTS. ")
            print("**************** Logged Summary ******************")
            Statistics.Print()
            print("**************** Logged Summary ******************")

    def launch_new_iteration(self) -> None:
        """
        Launch a new iteration and push some tasks to the task pool. 
        Tasks: Tree parallelization, Leaf parallelization
        """

        # Selecting a node
        logging.info("launching new iterations")
        rollout_result = self.mcts.do_rollout(self.mcts._sampler.root())
        
        if rollout_result is None:
            logging.info("The tree is exhausted......")
            while True:
                pass
            
        receipt, trials = rollout_result

        new_path: Dict[str, Tuple[TreeNode, Any]] = {}

        for path, node in trials:
            if not self.has_eval_result(node):
                new_path[path.serialize()] = (node, receipt)
            elif not self.check_dead(node):
                reward = self.get_eval_result(node)

                # update
                self.mcts.back_propagate(receipt, reward)
                self.reward_list.append(reward)
                self.time_list.append(time() - self.start_time)
            else:
                logging.warning("Encountered dead trial. Is that desired? ")

        return new_path

    def dump_result(self, result_save_loc: str = './final_result') -> None:
        """Search for the best model for iterations times."""

        print("Finish searching process. Displaying final result...")
        node, reward = self.best_node

        os.makedirs(result_save_loc, exist_ok=True)
        perf_path = os.path.join(result_save_loc, 'perf.json')
        perf_dict = {
            "rewards": self.reward_list,
            "times": self.time_list
        }
        json.dump(perf_dict, open(perf_path, 'w'))
        result_path = os.path.join(result_save_loc, 'result.json')
        result_dict = {
            "best_node": hash(node),
            "best_perf": reward,
            "mcts": self.mcts.serialize()
        }
        json.dump(result_dict, open(result_path, 'w'))

        assert self.has_eval_result(node)
        print("Best performance: {}".format(self.get_eval_result(node)))
        print("Time elapsed: {} seconds.".format(time() - self.start_time))
