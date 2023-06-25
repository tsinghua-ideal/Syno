import math
import torch

# Systems
import shutil
import os, sys
import json
import logging
from time import time, sleep
from typing import Dict, List, Tuple, Any
from copy import deepcopy

# KAS
from KAS import MCTS, Sampler, Statistics, TreePath, TreeNode, CodeGenOptions

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from utils.models import ModelBackup
import utils.models as model_factory
from utils.parser import arg_parse
from utils.config import parameters

class MCTSTrainer:
    def __init__(self,
                 sampler: Sampler,
                 arguments: Dict,
                 mcts_iterations: int = 1000,
                 leaf_parallelization_number: int = 5,
                 virtual_loss_constant: float = 5.,
                 exploration_weight: float = math.sqrt(2),
                 b: float = 0.4,
                 c_l: float = 40
                 ) -> None:
        
        self.mcts = MCTS(sampler, virtual_loss_constant,
                         leaf_parallelization_number, exploration_weight, b, c_l)

        print("MCTS initialized")
        self.remain_iterations = mcts_iterations
        self.start_time: float = time()

        # records
        self.best_node: TreeNode = None
        self.reward_list: List[float] = []
        self.time_list: List[float] = []

        # arguments
        self.arguments = arguments

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
        return self.arguments

    def update_result(self, meta: Tuple[TreeNode, Any], reward: float, path_to_trail: TreePath) -> None:
        """preprocess a path after evaluation. """
        trial, receipt = meta
        assert trial.is_final()
        if reward == -1:
            self.mcts.remove(receipt, trial)
        else:
            self.set_eval_result(trial, reward)
            # update
            if self.best_node is None or self.best_node.reward < reward:
                self.best_node = trial
            self.mcts.back_propagate(receipt, reward, path_to_trail)
            self.reward_list.append(reward)
            self.time_list.append(time() - self.start_time)
            print("Successfully updated MCTS. ")
            print("**************** Logged Summary ******************")
            Statistics.Print()
            print("**************** Logged Summary ******************")

    def launch_new_iteration(self) -> Dict[str, Tuple[TreeNode, Any]]:
        """
        Launch a new iteration and push some tasks to the task pool. 
        Tasks: Tree parallelization, Leaf parallelization
        """
        start = time()
        
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
                self.mcts.back_propagate(receipt, reward, path)
                self.reward_list.append(reward)
                self.time_list.append(time() - self.start_time)
            else:
                logging.warning("Encountered dead trial. Is that desired? ")

        logging.info("This iteration takes {} seconds".format(time() - start))
        return new_path

    def dump_result(self, result_save_loc: str = './final_result') -> None:
        """Search for the best model for iterations times."""

        print("Finish searching process. Displaying final result...")

        os.makedirs(result_save_loc, exist_ok=True)
        
        args_path = os.path.join(result_save_loc, 'arguments.json')
        # We need not to save mcts_iterations
        json.dump(self.arguments, open(args_path, 'w'), indent=4)
        
        perf_path = os.path.join(result_save_loc, 'perf.json')
        perf_dict = {
            "rewards": self.reward_list,
            "times": self.time_list
        }
        json.dump(perf_dict, open(perf_path, 'w'), indent=4)
        result_path = os.path.join(result_save_loc, 'result.json')
        result_dict = {
            "best_perf": self.best_node.reward,
            "mcts": self.mcts.serialize()
        }
        json.dump(result_dict, open(result_path, 'w'), indent=4)

        assert self.has_eval_result(self.best_node)
        assert self.get_eval_result(self.best_node) == self.best_node.reward
        print("Best performance: {}".format(self.best_node.reward))
        print("Time elapsed: {} seconds.".format(time() - self.start_time))
    
    @staticmethod
    def load(result_save_loc: str, mcts_iterations: int = 1000) -> 'MCTSTrainer':
        """
        Load a MCTSTrainer from a result folder.
        """
        args_path = os.path.join(result_save_loc, 'arguments.json')
        perf_path = os.path.join(result_save_loc, 'perf.json')
        result_path = os.path.join(result_save_loc, 'result.json')
        arguments = json.load(open(args_path))
        perf_dict = json.load(open(perf_path))
        result_dict = json.load(open(result_path))
        
        sampler_params = arguments['sampler_args']
        extra_args = arguments['extra_args']
        model_type = getattr(model_factory, extra_args['model_type'])
        _model = ModelBackup(model_type, torch.randn(
            extra_args["sample_input_shape"]), "cpu")
        sampler_params['autoscheduler'] = getattr(
            CodeGenOptions.AutoScheduler, sampler_params['autoscheduler'])
        sampler = Sampler(net=_model.create_instance(), **sampler_params)
        
        mcts_trainer = MCTSTrainer(
            sampler=sampler, 
            arguments=arguments, 
            mcts_iterations=mcts_iterations
        )
        mcts_trainer.mcts = MCTS.deserialize(serialized=result_dict["mcts"], sampler=sampler)
        mcts_trainer.reward_list = perf_dict["rewards"]
        mcts_trainer.time_list = perf_dict["times"]
        
        # get best_node
        for tree_node in mcts_trainer.mcts._treenode_store.values():
            if tree_node.is_final() and tree_node.reward == result_dict["best_perf"]:
                mcts_trainer.best_node = tree_node
                break
        assert mcts_trainer.best_node is not None, "The tree has no best node! (i.e., it does not contain any result). "
        
        return mcts_trainer

if __name__ == '__main__':
    args = arg_parse()
    use_cuda = torch.cuda.is_available()

    os.makedirs(args.kas_sampler_save_dir, exist_ok=True)

    training_params, sampler_params, extra_args = parameters(args)

    arguments = deepcopy(dict(
        sampler_args=sampler_params,
        train_args=training_params,
        extra_args=extra_args
    ))
    arguments['sampler_args']['autoscheduler'] = str(
        arguments['sampler_args']['autoscheduler'])[14:]  # HACK: serialize the enum

    model_type = getattr(model_factory, extra_args['model_type'])
    _model = ModelBackup(model_type, torch.randn(
        extra_args["sample_input_shape"]), "cpu")
    kas_sampler = Sampler(net=_model.create_instance(), **sampler_params)

    searcher = MCTSTrainer(
        kas_sampler,
        arguments,
        mcts_iterations=args.kas_iterations,
        leaf_parallelization_number=args.kas_leaf_parallelization_number,
        virtual_loss_constant=args.kas_tree_parallelization_virtual_loss_constant
    )
    
    result = searcher.launch_new_iteration()
    for path, trial in result.items():
        assert trial[0].is_final()
        searcher.update_result(trial, 0.9, TreePath.deserialize(path))
    searcher.dump_result('./test_save_folder')
    
    # test loading
    searcher_recover = MCTSTrainer.load('./test_save_folder')
    
    shutil.rmtree('./test_save_folder')
    print("[Passed] Loading Test. ")