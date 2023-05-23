import math

# Systems
import os
import json

# KAS
from KAS import MCTS, Sampler


class MCTSTrainer(MCTS):
    def __init__(self,
                 sampler: Sampler,
                 arguments: dict,
                 mcts_iterations: int = 1000,
                 leaf_parallelization_number: int = 5,
                 exploration_weight: float = math.sqrt(2)
                 ) -> None:
        super().__init__(sampler, exploration_weight)
        self.args = arguments

        self.best_node = (0, 0)
        self.reward_list = []
        self.time_list = []
        self.end_flag = False

        self.eval_result_cache = {}  # -1 stands for dead node
        self.pending_evaluate_cache = {}  # path -> node, receipt
        self.waiting_result_cache = {}
        self.remain_iterations = mcts_iterations
        self.leaf_parallelization_number = leaf_parallelization_number

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

    def update_result(self, path, reward=None) -> None:
        """preprocess a path after evaluation. """
        node, receipt = self.waiting_result_cache.pop(path)['meta']
        if reward is None:
            self.set_dead(node)
        else:
            self.set_eval_result(node, reward)
            # update
            if self.best_node[1] < reward:
                self.best_node = (node, reward)
            self.back_propagate(receipt, reward)
            self.reward_list.append(reward)

    def launch_new_iteration(self) -> None:
        """
        Launch a new iteration and push some tasks to the task pool. 
        Tasks: Tree parallelization, Leaf parallelization
        """
        if self.remain_iterations == 0:
            self.end_flag = True
            return

        self.remain_iterations -= 1

        # Selecting a node
        # TODO: add virtual loss
        for _ in range(self.leaf_parallelization_number):
            receipt, node = self.do_rollout(self._sampler.root())
            while self.check_dead(node):
                receipt, node = self.do_rollout(self._sampler.root())

            if not self.has_eval_result(node):
                self.pending_evaluate_cache[node.path] = (node, receipt)
            else:
                reward = self.get_eval_result(node)

                # update
                if self.best_node[1] < reward:
                    self.best_node = (node, reward)
                self.back_propagate(receipt, reward)
                self.reward_list.append(reward)

        self.end_flag = False

    def dump_result(self, result_save_loc: str = './final_result') -> None:
        """Search for the best model for iterations times."""

        print("Finish searching process. Displaying final result...")

        os.makedirs(result_save_loc, exist_ok=True)
        perf_path = os.path.join(result_save_loc, 'perf.json')
        perf_dict = {"rewards": self.reward_list,
                     "times": self.time_list}
        json.dump(perf_dict, open(perf_path, 'w'))

        node = self.get_results()
        print("Best performance: {}".format(self.get_eval_result(node)))
