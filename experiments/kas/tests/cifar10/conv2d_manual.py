import torch

# Systems
import time
import random
import os
import sys
import logging

# KAS
from KAS import Sampler, Assembled, Assembler
from KAS.Bindings import CodeGenOptions


if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from train import train
from utils.data import get_dataloader
from utils.models import KASConv, ModelBackup, resnet8
from utils.parser import arg_parse
from utils.config import parameters


def conv2d(assembler: Assembler) -> Assembled:
    N, H, W, k, C_in, C_out = assembler.get_sizes(
        "N", "H", "W", "k_1", "C_in", "C_out")

    # Inputs: [N, H, W], [C_out, k_1, k_2]
    in_N, in_H, in_W, in_C, out_C, w_in_C, w_k_1, w_k_2 = assembler.make_dims_of_sizes(
        N, H, W, C_in, C_out, C_in, k, k)
    # [in_N, in_H, in_W, in_C, out_C, w_in_C, w_k_1, w_k_2]

    main_H, windows_H = assembler.create_unfold(in_H, k)
    main_W, windows_W = assembler.create_unfold(in_W, k)
    # [in_N, main_H, windows_H, main_W, windows_W, in_C, out_C, w_in_C, w_k_1, w_k_2]

    shared_k_1 = assembler.create_share(windows_H, w_k_1)
    shared_k_2 = assembler.create_share(windows_W, w_k_2)
    shared_C_in = assembler.create_share(in_C, w_in_C)
    # [in_N, main_H, main_W, shared_C_in, out_C, shared_k_1, shared_k_2]

    in_N.output(0)
    out_C.output(1)
    main_H.output(2)
    main_W.output(3)
    shared_k_1.mean(0)
    shared_k_2.mean(1)
    shared_C_in.mean(2)

    return assembler.assemble('in_0 * in_1', [in_N, in_C, in_H, in_W], [out_C, w_in_C, w_k_1, w_k_2])


if __name__ == '__main__':

    start = time.time()

    # set logging level
    logging.getLogger().setLevel(logging.INFO)

    args = arg_parse()
    assert args.dataset == 'cifar10'
    training_params, sampler_params, extra_args = parameters(args)
    use_cuda = torch.cuda.is_available()

    os.makedirs(args.kas_sampler_save_dir, exist_ok=True)

    train_data_loader, validation_data_loader = get_dataloader(args)

    device = torch.device("cuda" if use_cuda else "cpu")
    model_ = ModelBackup(KASConv, torch.randn(
        extra_args["sample_input_shape"]), device)

    kas_sampler = Sampler(
        net=model_.create_instance(),
        **sampler_params
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
        model, train_data_loader, validation_data_loader, args, **training_params, verbose=True)
    accuracy = 1. - min(val_error)

    print("Test Complete, elapsed {} seconds, accuracy {}. ".format(
        time.time() - start, accuracy))
