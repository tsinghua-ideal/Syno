import torch
from torch import nn, Tensor

import math

# Systems
import time
import random
from typing import List, Union
import os
import sys
import logging
import traceback
import json
from thop import profile

# KAS
from KAS import MCTS, Sampler, KernelPack, Node, Path, Placeholder
from KAS.Bindings import CodeGenOptions

from train import train
from utils.data import get_dataloader
from utils.models import KASGrayConv as KASConv, ModelBackup
from utils.parser import arg_parse


class MCTSTrainer(MCTS):
    def __init__(self,
                 model: ModelBackup,
                 train_params: dict,
                 sampler: Sampler,
                 typ: str,
                 exploration_weight: float = math.sqrt(2),
                 min_model_size=0,
                 max_model_size=0.3,
                 min_macs=0,
                 max_macs=1
                 ) -> None:
        super().__init__(sampler, exploration_weight)
        self._train_params = train_params
        self._device = torch.device(
            "cuda" if self._train_params["use_cuda"] else "cpu")
        self._model = model

        self.min_model_size = int(min_model_size * 1e6)
        self.max_model_size = int(max_model_size * 1e6)
        self.min_macs = int(min_macs * 1e9)
        self.max_macs = int(max_macs * 1e9)

        self.type = typ  # 'mcts' or 'random'
        self.best_node = (0, 0)
        self.reward_list = []
        self.time_list = []
        self.search_start_timestamp = 0

        self.eval_result_cache = {}  # -1 stands for dead node
        self.pending_evaluate_cache = {}  # path -> node, receipt*,

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

    def update_result(self, path, reward) -> None:
        if self.type == 'mcts':
            node, receipt = self.pending_evaluate_cache[path]
        else:
            node = self.pending_evaluate_cache[path]
        self.set_eval_result(node, reward)
        # update
        if self.best_node[1] < reward:
            self.best_node = (node, reward)
        if self.type == 'mcts':
            self.back_propagate(receipt, reward)
        self.reward_list.append(reward)

    def update(self, prefix="") -> str:

        # Selecting a node
        if self.type == 'mcts':
            receipt, node = self.do_rollout(kas_sampler.root())
            while self.check_dead(node):
                receipt, node = self.do_rollout(kas_sampler.root())
        elif self.type == 'random':
            while True:
                node = self._sampler.random_node_with_prefix(Path([]))
                if node.is_final():
                    break

        if not self.has_eval_result(node):
            # Analyze a sample
            model = self._model.create_instance()

            # Early filter MACs
            print(f"Estimated flops {node.estimate_total_flops_as_final()}")
            model_macs_estimated = self._model.base_macs + \
                node.estimate_total_flops_as_final() * 2
            if model_macs_estimated > self.max_macs * 10:
                print(
                    f"(Estimated) Model macs {model_macs_estimated} exceeds {self.max_macs} * 10, skipping")
                self.set_dead(node)
                return "FAILED because of too big model macs."
            elif model_macs_estimated < self.min_macs * 0.1:
                print(
                    f"(Estimated) Model macs {model_macs_estimated} below {self.min_macs} * 0.1, skipping")
                self.set_dead(node)
                return "FAILED because of too small model macs."

            kernelPacks, total_flops = self._sampler.realize(
                model, node, prefix)
            model = self._model.restore_model_params(model, kernelPacks)

            model_macs = self._model.base_macs + total_flops * 2
            print("Model macs: {}".format(model_macs))
            if model_macs > self.max_macs:
                print(
                    f"Model macs {model_macs} exceeds {self.max_macs}, skipping")
                self.set_dead(node)
                return "FAILED because of too big model macs."
            elif model_macs < self.min_macs:
                print(
                    f"Model macs {model_macs} below {self.min_macs}, skipping")
                self.set_dead(node)
                return "FAILED because of too small model macs."

            model_size = sum([p.numel() for p in model.parameters()])
            print("Model size: {}".format(model_size))
            if model_size > self.max_model_size:
                print(
                    f"Model size {model_size} exceeds {self.max_model_size}, skipping")
                self.set_dead(node)
                return "FAILED because of too big model size."
            elif model_size < self.min_model_size:
                print(
                    f"Model size {model_size} below {self.min_model_size}, skipping")
                self.set_dead(node)
                return "FAILED because of too small model size."

            reward = self._eval_model(model)
            self.set_eval_result(node, reward)

        assert self.has_eval_result(node) and not self.check_dead(node)
        reward = self.get_eval_result(node)

        # update
        if self.best_node[1] < reward:
            self.best_node = (node, reward)
        if self.type == 'mcts':
            self.back_propagate(receipt, reward)

        self.reward_list.append(reward)
        self.time_list.append(time.time() - self.search_start_timestamp)

        return "SUCCESS"

    def search(self, iterations: int = 1000, result_save_loc: str = './final_result') -> None:
        """Search for the best model for iterations times."""
        self.search_start_timestamp = time.time()
        for iter in range(iterations):
            print(f"Iteration {iter}")
            while True:
                try:
                    # Now the iteration end even when it fails
                    result = self.update(self.type + f"Iteration{iter}")
                    print(f"Iteration {iter} {result}.")
                    if result == "SUCCESS":
                        break
                except Exception as e:
                    logging.warning("Catched error {}, retrying".format(e))
                    traceback.print_exc(file=sys.stderr)

        print("Finish searching process. Displaying final result...")

        os.makedirs(result_save_loc, exist_ok=True)
        perf_path = os.path.join(result_save_loc, 'perf.json')
        perf_dict = {"rewards": self.reward_list,
                     "times": self.time_list}
        json.dump(perf_dict, open(perf_path, 'w'))

        if self.type == 'mcts':
            node = self.get_results()
        elif self.type == 'random':
            node = self.best_node[0]

        model = self._model.create_instance()
        kernelPacks, _ = self._sampler.realize(model, node, "FinalResult")
        model = self._model.restore_model_params(model, kernelPacks)
        train_error, val_error, best_model_state_dict = train(
            model, **self._train_params, verbose=True)

        model_ckpt = (node.path.serialize(), best_model_state_dict)
        model_path = os.path.join(result_save_loc, 'model.pth')
        torch.save(model_ckpt, model_path)

        print("The best model is saved in {}".format(model_path))
        print("Best performance: {}".format(1. - min(val_error)))


if __name__ == '__main__':

    start = time.time()

    # set logging level
    logging.getLogger().setLevel(logging.INFO)

    args = arg_parse()
    use_cuda = torch.cuda.is_available()

    os.makedirs(args.kas_sampler_save_dir, exist_ok=True)

    training_params = dict(
        criterion=nn.CrossEntropyLoss(),
        lr=0.1,
        momentum=0.9,
        epochs=30,
        val_period=5,
        use_cuda=use_cuda
    )

    sampler_params = dict(
        input_shape="[N,H,W]",
        output_shape="[N,C_out,H,W]",
        primary_specs=["N=4096: 1", "H=256", "W=256", "C_out=100"],
        coefficient_specs=["s_1=2", "k_1=3", "k_2=5"],
        seed=random.SystemRandom().randint(
            0, 0x7fffffff) if args.kas_seed == 'pure' else args.seed,
        depth=args.kas_depth,
        dim_lower=args.kas_min_dim,
        dim_upper=args.kas_max_dim,
        save_path=args.kas_sampler_save_dir,
        cuda=use_cuda,
        fixed_io_pairs=[(0, 0)],
        autoscheduler=CodeGenOptions.AutoScheduler.Anderson2021
    )

    # TODO: Set up 8 evaluators.

    extra_args = dict(
        max_macs=args.kas_max_macs,
        min_macs=args.kas_min_macs,
        max_model_size=args.kas_max_params,
        min_model_size=args.kas_min_params,
        prefix="",
        model_type="KASConv",  # TODO: dynamically load the module
        batch_size=args.batch_size,
        device=torch.device("cuda" if use_cuda else "cpu")
    )

    _model = ModelBackup(KASConv, torch.randn(
        extra_args["sample_input_shape"]), extra_args["device"])
    kas_sampler = Sampler(net=_model.create_instance(), **sampler_args)

    searcher = MCTSTrainer(model, training_params,
                           kas_sampler, args.kas_searcher_type, min_model_size=args.kas_min_params, max_model_size=args.kas_max_params, min_macs=args.kas_min_macs, max_macs=args.kas_max_macs)

    searcher.search(iterations=args.kas_iterations,
                    result_save_loc=args.result_save_dir)

    print("Search Complete, elapsed {} seconds. ".format(
        time.time() - start))
