import torch

# Systems
import time
import os
import sys
import logging

# KAS
from KAS import Sampler, Assembled, Assembler


if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from train import train
from utils.data import get_dataloader
from utils.models import KASFC, ModelBackup
from utils.parser import arg_parse
from utils.config import parameters


def dense(assembler: Assembler) -> Assembled:
    N, C_in, C_out = assembler.get_sizes(
        "N", "C_in", "C_out")

    # Inputs: [N, C_in], [C_in, C_out]
    in_N, in_C, w_in_C, w_out_C = assembler.make_dims_of_sizes(
        N, C_in, C_in, C_out)
    # [in_N, in_C, w_in_C, w_out_C]

    shared_C_in = assembler.create_share(in_C, w_in_C)
    # [in_N, shared_C_in, w_out_C]

    in_N.output(0)
    w_out_C.output(1)
    shared_C_in.mean(0)

    return assembler.assemble([in_N, in_C], [w_in_C, w_out_C])


if __name__ == '__main__':

    start = time.time()

    # set logging level
    logging.getLogger().setLevel(logging.INFO)

    args = arg_parse()
    training_params, sampler_params, extra_args = parameters(args)
    use_cuda = torch.cuda.is_available()

    os.makedirs(args.kas_sampler_save_dir, exist_ok=True)

    train_data_loader, validation_data_loader = get_dataloader(args)

    device = torch.device("cuda" if use_cuda else "cpu")
    model_ = ModelBackup(KASFC, torch.randn(
        extra_args["sample_input_shape"]), device)

    kas_sampler = Sampler(
        net=model_.create_instance(),
        **sampler_params
    )

    assembler = kas_sampler.create_assembler()
    assembled = dense(assembler)
    print("The path is", assembled.convert_to_path(kas_sampler))

    # Analyze a sample
    model = model_.create_instance()

    kernelPacks, total_flops = kas_sampler.realize(
        model, assembled, "test_manual_fc")
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
