import torch
from torch import nn

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


class model_backup:
    def __init__(self, model: nn.Module):
        self.model = deepcopy(model).cpu()

    def restore_model_params(self, pack: KernelPack, device='cuda:0'):
        """
        Restore model parameters and replace the selected parameters with pack.
        """
        model = deepcopy(self.model).to(device)
        placeholders = Modifier.FindPlaceholders(model)
        if len(pack) > 0:
            assert isinstance(pack[0], KernelPack
                              ), f"elements in pack are not valid! {type(pack[0])}"
            Modifier.KernelReplace(placeholders, pack)
        return model


class Searcher(MCTS):
    def __init__(self,
                 model: nn.Module, train_params: dict,
                 sampler: Sampler, exploration_weight: float = math.sqrt(2), save_path: str = './search_result') -> None:
        super().__init__(sampler, exploration_weight)
        assert model.bootstraped(), "The model is not yet replaced by KAS!"
        self._model = model_backup(model)
        self._save_path = save_path
        self._train_params = train_params

    def generate_kernel(self, model: nn.Module, path: List[int]) -> List[KernelPack]:
        """Generate a kernel. """
        kernel = self._sampler._realize(path)
        logging.debug(f"Path: {self._sampler._path_str(path)}")
        logging.debug(f"Kernel: {kernel}")

        identifier_prefix = '_'.join(map(str, path))
        save_path = os.path.join(self._save_path, identifier_prefix)
        kernelPacks = []
        os.makedirs(save_path, exist_ok=True)
        for i, placeholder in enumerate(Modifier.FindPlaceholders(model)):
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

    def update(self) -> None:
        path = self.do_rollout([])
        kernelPacks = self.generate_kernel(self._model.model, path)
        model = self._model.restore_model_params(kernelPacks)
        reward = self._eval_model(model)
        self.back_propagate(path, reward)

    def search(self, iterations=1000):
        for iter in range(iterations):
            print(f"Iteration {iter}")
            self.update()


if __name__ == '__main__':

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    args = arg_parse()

    train_data_loader, validation_data_loader = get_dataloader(args)

    model = KASConv()
    model.generatePlaceHolder(
        train_data_loader.dataset[0][0][None, :].repeat(args.batch_size, 1, 1, 1))

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

    searcher = Searcher(model, training_params, kas_sampler,
                        save_path=args.kas_searcher_save_dir)

    searcher.search(iterations=args.kas_iterations)

    print("Searched Ended:")
    path = searcher.get_results()
    kernelPacks = searcher.generate_kernel(path)
    model = searcher._model.restore_model_params(kernelPacks)
    train_error, val_error = train(model, **training_params, verbose=True)
