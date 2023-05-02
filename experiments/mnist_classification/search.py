import torch
from torch import nn, Tensor

import math

# Systems
import time
import random
from typing import List
import os
import sys
import logging
import traceback
import json

# KAS
from KAS import MCTS, Sampler, KernelPack, Node, Path
from kas_cpp_bindings import CodeGenOptions

from train import train
from utils.data import get_dataloader
from utils.models import KASGrayConv as KASConv
from utils.parser import arg_parse


class ModelBackup:
    def __init__(self, model, sample_input: Tensor, device='cuda:0') -> None:
        self._model_builder = model
        self._sample_input = sample_input.to(device)

        self._model = self._model_builder().to(device)
        self._model.generatePlaceHolder(self._sample_input)

    def create_instance(self) -> nn.Module:
        self._model._initialize_weight()
        return self._model

    def restore_model_params(self, model, pack: List[KernelPack]):
        """
        Restore model parameters and replace the selected parameters with pack.
        """
        assert len(pack) > 0, "Not detected any placeholders! "
        assert isinstance(pack[0], KernelPack
                          ), f"elements in pack are not valid! {type(pack[0])}"
        Sampler.replace(model, pack)
        return model


class Searcher(MCTS):
    def __init__(self,
                 model: nn.Module, sample_input: Tensor, train_params: dict,
                 sampler: Sampler, typ: str, exploration_weight: float = math.sqrt(2), min_model_size=0.1, max_model_size=0.3) -> None:
        super().__init__(sampler, exploration_weight)
        self._train_params = train_params
        self._device = torch.device(
            "cuda" if self._train_params["use_cuda"] else "cpu")
        self._model = ModelBackup(model, sample_input, self._device)
        self.min_model_size = int(min_model_size * 1e6)
        self.max_model_size = int(max_model_size * 1e6)
        self.type = typ  # 'mcts' or 'random'
        self.best_node = None
        self.upper_bound_list = []

    def _eval_model(self, model: nn.Module) -> float:
        train_error, val_error, _ = train(
            model, **self._train_params, verbose=True)
        accuracy = 1. - min(val_error)
        assert 0 <= accuracy <= 1
        return accuracy

    def update(self, prefix="") -> str:

        # Selecting a node
        if self.type == 'mcts':
            node = self.do_rollout(kas_sampler.root())
        elif self.type == 'random':
            while True:
                node = self._sampler.random_node_with_prefix(Path([]))
                if node.is_final():
                    break

        # Analyze a sample
        model = self._model.create_instance()
        kernelPacks = self._sampler.realize(model, node, prefix)
        model = self._model.restore_model_params(model, kernelPacks)
        model_size = sum(p.numel()
                         for p in model.parameters() if p.requires_grad)
        logging.info("Model size: {}".format(model_size))

        if model_size > self.max_model_size:
            logging.info(
                f"Model size {model_size} exceeds {self.max_model_size}, skipping")
            return "FAILED because of too big model size."
        elif model_size < self.min_model_size:
            logging.info(
                f"Model size {model_size} below {self.min_model_size}, skipping")
            return "FAILED because of too small model size."

        reward = self._eval_model(model)

        # update
        if self.best_node is None or self.best_node[1] < reward:
            self.best_node = (node, reward)
        if self.type == 'mcts':
            self.back_propagate(node, reward)

        return "SUCCESS"

    def search(self, iterations: int = 1000, result_save_loc: str = './final_result') -> None:
        """Search for the best model for iterations times."""
        for iter in range(iterations):
            logging.info(f"Iteration {iter}")
            while True:
                try:
                    result = self.update(self.type + f"Iteration{iter}")
                    self.upper_bound_list.append(self.best_node[1])
                    logging.info(f"Iteration {iter} {result}.")
                except Exception as e:
                    logging.warning("Catched error {}, retrying".format(e))
                    traceback.print_exc(file=sys.stderr)

        logging.info("Finish searching process. Displaying final result...")

        os.makedirs(result_save_loc, exist_ok=True)
        perf_path = os.path.join(result_save_loc, 'perf.json')
        json.dump(self.upper_bound_list, open(perf_path, 'w'))

        if self.type == 'mcts':
            node = self.get_results()
        elif self.type == 'random':
            node = self.best_node[0]

        model = self._model.create_instance()
        kernelPacks = self._sampler.realize(model, node, "FinalResult")
        model = self._model.restore_model_params(model, kernelPacks)
        train_error, val_error, best_model_state_dict = train(
            model, **self.training_params, verbose=True)

        model_ckpt = (node.path.serialize(), best_model_state_dict)
        model_path = os.path.join(result_save_loc, 'model.pth')
        torch.save(model_ckpt, model_path)

        logging.info("The best model is saved in {}".format(model_path))
        logging.info("Best performance: {}".format(min(val_error)))


if __name__ == '__main__':

    # set logging level
    logging.getLogger().setLevel(logging.INFO)

    args = arg_parse()
    use_cuda = torch.cuda.is_available()

    # if os.path.exists(args.kas_sampler_save_dir):
    #     shutil.rmtree(args.kas_sampler_save_dir)
    os.makedirs(args.kas_sampler_save_dir, exist_ok=True)

    train_data_loader, validation_data_loader = get_dataloader(args)

    sample_input = train_data_loader.dataset[0][0][None, :].repeat(
        args.batch_size, 1, 1, 1)

    training_params = dict(
        train_loader=train_data_loader,
        val_loader=validation_data_loader,
        criterion=nn.CrossEntropyLoss(),
        lr=0.1,
        momentum=0.9,
        epochs=30,
        val_period=5,
        use_cuda=use_cuda
    )

    kas_sampler = Sampler(
        input_shape="[N,H,W]",  # "[N,C_in,H,W]",
        output_shape="[N,C_out,H,W]",
        primary_specs=["N=128: 1", "H=256", "W=256", "C_out=100"],
        coefficient_specs=["s_1=2: 2", "k_1=3", "5"],
        seed=random.SystemRandom().randint(
            0, 0x7fffffff) if args.kas_seed == 'pure' else args.seed,
        depth=args.kas_depth,
        dim_lower=args.kas_min_dim,
        dim_upper=args.kas_max_dim,
        save_path=args.kas_sampler_save_dir,
        cuda=use_cuda,
        autoscheduler=CodeGenOptions.AutoScheduler.Anderson2021
    )

    searcher = Searcher(KASConv, sample_input, training_params,
                        kas_sampler, args.kas_searcher_type, min_model_size=args.kas_min_params, max_model_size=args.kas_max_params)

    searcher.search(iterations=args.kas_iterations,
                    result_save_loc=args.result_save_dir)
