import torch
from torch import nn, Tensor

import math

# Systems
import time
import random
from copy import deepcopy
from typing import List
import os
import sys
import logging

# KAS
from KAS import MCTS, Sampler, Modifier, KernelPack
from kas_cpp_bindings import CodeGenOptions

from train import train
from utils.data import get_dataloader
from utils.models import KASConv
from utils.parser import arg_parse


class ModelBackup:
    def __init__(self, model, sample_input: Tensor) -> None:
        self._model_builder = model
        self._sample_input = sample_input

    def create_instance(self) -> nn.Module:
        model = self._model_builder()
        model.generatePlaceHolder(self._sample_input.clone())
        return model

    def restore_model_params(self, model, pack: List[KernelPack], device='cuda:0'):
        """
        Restore model parameters and replace the selected parameters with pack.
        """
        placeholders = Modifier.find_placeholders(model)
        if len(pack) > 0:
            assert isinstance(pack[0], KernelPack
                              ), f"elements in pack are not valid! {type(pack[0])}"
            Modifier.kernel_replace(placeholders, deepcopy(pack))
        model = model.to(device)
        return model


class Searcher(MCTS):
    def __init__(self,
                 model: nn.Module, sample_input: Tensor, train_params: dict,
                 sampler: Sampler, exploration_weight: float = math.sqrt(2), save_path: str = './search_result') -> None:
        super().__init__(sampler, exploration_weight)
        self._model = ModelBackup(model, sample_input)
        self._save_path = save_path
        self._train_params = train_params

    def generate_kernel(self, model: nn.Module, path: List[int], prefix: str) -> List[KernelPack]:
        """Generate a kernel. """
        kernel = self._sampler._realize(path)
        logging.debug(f"Path: {self._sampler._path_str(path)}")
        logging.debug(f"Kernel: {kernel}")

        identifier_prefix = prefix + '_' + '_'.join(map(str, path))
        save_path = os.path.join(self._save_path, identifier_prefix)
        kernelPacks = []
        os.makedirs(save_path, exist_ok=True)
        for i, placeholder in enumerate(Modifier.find_placeholders(model)):
            kernel_name = f'kernel_{i}'
            mappings = placeholder.mappings
            logging.debug(f"For kernel_{i} mappings: {mappings}")
            kernel.generate(save_path, kernel_name, mappings)
            identifier = identifier_prefix + "__" + str(i)
            inputs_shapes = kernel.get_inputs_shapes(mappings)
            logging.debug(f"Inputs shapes: {inputs_shapes}")
            output_shape = kernel.get_output_shape(mappings)
            logging.debug(f"Output shape: {output_shape}")
            kernelPacks.append(
                KernelPack(identifier, save_path, kernel_name,
                           inputs_shapes, output_shape, self._sampler._device)
            )
        return kernelPacks

    def _eval_model(self, model: nn.Module) -> float:
        train_error, val_error = train(
            model, **self._train_params, verbose=True)
        accuracy = 1. - min(val_error)
        assert 0 <= accuracy <= 1
        return accuracy

    def update(self, prefix="") -> None:
        path = self.do_rollout([])
        model = self._model.create_instance()
        kernelPacks = self.generate_kernel(model, path, prefix)
        model = self._model.restore_model_params(model, kernelPacks)
        reward = self._eval_model(model)
        self.back_propagate(path, reward)

    def search(self, iterations=1000):
        for iter in range(iterations):
            print(f"Iteration {iter}")
            self.update(f"Iteration{iter}")


if __name__ == '__main__':

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    args = arg_parse()

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
        val_period=1,
        use_cuda=torch.cuda.is_available()
    )

    kas_sampler = Sampler(
        input_shape="[N,C_in,H,W]",
        output_shape="[N,C_out,H,W]",
        primary_specs=[],
        coefficient_specs=["s_1=2: 2", "k_1=3", "5"],
        seed=random.SystemRandom().randint(
            0, 0x7fffffff) if args.kas_seed == 'pure' else args.seed,
        depth=args.kas_depth,
        dim_lower=args.kas_min_dim,
        dim_upper=args.kas_max_dim,
        save_path=args.kas_sampler_save_dir,
        cuda=torch.cuda.is_available(),
        autoscheduler=CodeGenOptions.AutoScheduler.Anderson2021
    )

    searcher = Searcher(KASConv, sample_input, training_params, kas_sampler,
                        save_path=args.kas_searcher_save_dir)

    searcher.search(iterations=args.kas_iterations)

    print("Searched Ended:")
    path = searcher.get_results()
    model = searcher._model.create_instance()
    kernelPacks = searcher.generate_kernel(model, path, "FinalResult")
    model = searcher._model.restore_model_params(model, kernelPacks)
    train_error, val_error = train(model, **training_params, verbose=True)
