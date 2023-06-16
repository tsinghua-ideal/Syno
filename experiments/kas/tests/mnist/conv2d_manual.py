import torch
from torch import nn, Tensor

# Systems
import time
import random
from typing import List
import os
import sys
import logging
from thop import profile

# KAS
from KAS import Sampler, KernelPack, Assembled, Assembler
from KAS.Bindings import CodeGenOptions


if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from train import train
from utils.data import get_dataloader
from utils.models import KASGrayConv as KASConv
from utils.parser import arg_parse


class ModelBackup:
    def __init__(self, model, sample_input: Tensor, device='cuda:0') -> None:
        self._model_builder = model
        self._sample_input = sample_input.to(device)

        self._model = self._model_builder().to(device)
        macs, params = profile(self._model, inputs=(
            self._sample_input, ), verbose=False)
        print(
            f"Referenced model has {round(macs / 1e9, 3)}G MACs and {round(params / 1e6, 3)}M parameters ")
        self.base_macs = macs - \
            self._model.generatePlaceHolder(self._sample_input)
        print(f"Base MACs is {self.base_macs}")

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


def conv2d(assembler: Assembler) -> Assembled:
    N, H, W, k, C_out = assembler.get_sizes(
        "N", "H", "W", "k_2", "C_out")

    # Inputs: [N, H, W], [C_out, k_1, k_2]
    in_N, in_H, in_W, out_C, w_k_1, w_k_2 = assembler.make_dims_of_sizes(
        N, H, W, C_out, k, k)
    # [in_N, in_H, in_W, out_C, w_k_1, w_k_2]

    main_H, windows_H = assembler.create_unfold(in_H, k)
    main_W, windows_W = assembler.create_unfold(in_W, k)
    # [in_N, main_H, windows_H, main_W, windows_W, out_C, w_k_1, w_k_2]

    shared_k_1 = assembler.create_share(windows_H, w_k_1)
    shared_k_2 = assembler.create_share(windows_W, w_k_2)
    # [in_N, main_H, main_W, out_C, shared_k_1, shared_k_2]

    in_N.output(0)
    out_C.output(1)
    main_H.output(2)
    main_W.output(3)
    shared_k_1.sum(0)
    shared_k_2.sum(1)

    return assembler.assemble([in_N, in_H, in_W], [out_C, w_k_1, w_k_2])


if __name__ == '__main__':

    start = time.time()

    # set logging level
    logging.getLogger().setLevel(logging.INFO)

    args = arg_parse()
    use_cuda = torch.cuda.is_available()

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

    device = torch.device("cuda" if use_cuda else "cpu")
    model_ = ModelBackup(KASConv, sample_input, device)

    kas_sampler = Sampler(
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
        net=model_.create_instance(),
        autoscheduler=CodeGenOptions.AutoScheduler.Anderson2021
    )

    assembler = kas_sampler.create_assembler()

    # Analyze a sample
    model = model_.create_instance()

    kernelPacks, total_flops = kas_sampler.realize(
        model, conv2d(assembler), "test_manual_conv")
    model = model_.restore_model_params(model, kernelPacks)

    model_macs = model_.base_macs + total_flops * 2
    print("Model macs: {}".format(model_macs))
    model_size = sum([p.numel() for p in model.parameters()])
    print("Model size: {}".format(model_size))

    train_error, val_error, _ = train(
        model, **training_params, verbose=True)
    accuracy = 1. - min(val_error)

    print("Test Complete, elapsed {} seconds, accuracy {}. ".format(
        time.time() - start, accuracy))
