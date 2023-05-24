import math

# Systems
import os
import json
from time import time, sleep

# KAS
from KAS import MCTS, Sampler


class MCTSTrainer(MCTS):
    def __init__(self,
                 sampler: Sampler,
                 arguments: dict,
                 mcts_iterations: int = 1000,
                 leaf_parallelization_number: int = 5,
                 virtual_loss_constant: float = 5.,
                 simulate_retry_limit: int = 10,
                 exploration_weight: float = math.sqrt(2)
                 ) -> None:
        super().__init__(sampler, virtual_loss_constant,
                         leaf_parallelization_number, simulate_retry_limit, exploration_weight)

        # flags
        self.best_node = (0, 0)
        self.reward_list = []
        self.time_list = []
        self.end_flag = False
        self.start_time = time()

        # arguments
        self.args = arguments
        self.remain_iterations = mcts_iterations
        self.leaf_parallelization_number = leaf_parallelization_number
        self.virtual_loss_constant = virtual_loss_constant

        # buffer
        self.eval_result_cache = {}  # node -> reward; -1 stands for dead node
        self.pending_evaluate_cache = {}  # path -> trial, receipt
        self.waiting_result_cache = {}  # path -> trial, receipt

    def has_eval_result(self, node) -> bool:
        return node._node in self.eval_result_cache

    def check_dead(self, node) -> bool:
        return self.has_eval_result(node) and self.eval_result_cache[node._node] == -1

    def get_eval_result(self, node) -> float:
        assert not self.check_dead(node), "Node is dead!"
        return self.eval_result_cache[node._node]

    def set_dead(self, node) -> None:
        self.eval_result_cache[node._node] = -1

    def set_eval_result(self, node, val: float) -> None:
        assert 0 <= val <= 1, "Invalid eval result!"
        assert not self.has_eval_result(node), "Node already has eval result!"
        self.eval_result_cache[node._node] = val

    def get_args(self):
        return self.args

    def update_result(self, path, reward) -> None:
        """preprocess a path after evaluation. """
        node, receipt = self.waiting_result_cache.pop(path)['meta']
        if reward == -1:
            self.set_dead(node)
        else:
            self.set_eval_result(node, reward)
            # update
            if self.best_node[1] < reward:
                self.best_node = (node, reward)
            self.back_propagate(receipt, reward)
            self.reward_list.append(reward)
            self.time_list.append(time() - self.start_time)

    def launch_new_iteration(self) -> None:
        """
        Launch a new iteration and push some tasks to the task pool. 
        Tasks: Tree parallelization, Leaf parallelization
        """
        if self.remain_iterations == 0:
            if len(self.waiting_result_cache) == 0:
                self.end_flag = True
            else:
                print(f"Waiting for the following tasks to be done: {list(self.waiting_result_cache.keys())}")
                sleep(60) # recheck every minute
            return

        self.remain_iterations -= 1
        print(f"Remaining iterations: {self.remain_iterations}")

        # Selecting a node
        # TODO: add virtual loss
        receipt, trials = self.do_rollout(self._sampler.root())
        while any([self.check_dead(trial) for trial in trials]):
            receipt, trials = self.do_rollout(self._sampler.root())

        for trial in trials:
            if not self.has_eval_result(trial):
                self.pending_evaluate_cache[trial.path] = (trial, receipt)
            else:
                reward = self.get_eval_result(trial)

                # update
                self.back_propagate(receipt, reward)
                self.reward_list.append(reward)
                self.time_list.append(time() - self.start_time)

    def dump_result(self, result_save_loc: str = './final_result') -> None:
        """Search for the best model for iterations times."""

        print("Finish searching process. Displaying final result...")
        node = self.get_results()

        os.makedirs(result_save_loc, exist_ok=True)
        perf_path = os.path.join(result_save_loc, 'perf.json')
        perf_dict = {
            "rewards": self.reward_list,
            "times": self.time_list
        }
        json.dump(perf_dict, open(perf_path, 'w'))
        result_path = os.path.join(result_save_loc, 'result.json')
        result_dict = {
            "best_path": node.path.serialize(),
            "mcts": self.dump()
        }
        json.dump(result_dict, open(result_path, 'w'))

        if self.has_eval_result(node):
            print("Best performance: {}".format(self.get_eval_result(node)))
        else:
            print(
                "[Warning] best path not evaluated. Consider running for more iterations. ")
