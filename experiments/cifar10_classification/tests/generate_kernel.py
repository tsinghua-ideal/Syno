"""
Parse the log of servers. Useful before the training ends. 
"""

import torch

import argparse
import re
import random
import os
import sys
from tqdm.contrib import tzip

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from utils.parser import arg_parse
from mp_utils import Handler_client
from utils.models import KASConv, ModelBackup
from tests.conv2d_manual import conv2d

from KAS import Sampler, TreePath
from KAS.Bindings import CodeGenOptions


def arg_parse_logger():
    parser = argparse.ArgumentParser(
        description='Graph Generator')

    parser.add_argument('--output_path', type=str,
                        default='./tests/', help='Path to the output')

    args = parser.parse_args()
    return args


def create_sample(args_logger):
    args = arg_parse()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    _model = ModelBackup(KASConv, torch.randn(
        (args.batch_size, 3, 32, 32)), device)

    sampler_args = dict(
        input_shape="[N,C_in,H,W]",
        output_shape="[N,C_out,H,W]",
        primary_specs=["N=64: 0", "H=32", "W=32"],
        coefficient_specs=["k_1=3: 4", "k_2=5: 4"],
        fixed_io_pairs=[(0, 0)],
        seed=random.SystemRandom().randint(
            0, 0x7fffffff) if args.kas_seed == 'pure' else args.seed,
        depth=args.kas_depth,
        dim_lower=args.kas_min_dim,
        dim_upper=args.kas_max_dim,
        maximum_tensors=2,
        maximum_reductions=4,
        max_flops=int(args.kas_max_macs * 1e9),
        save_path=os.path.join(args_logger.output_path, 'samples'),
        cuda=use_cuda,
        net=_model.create_instance(),
        autoscheduler=CodeGenOptions.AutoScheduler.Anderson2021
    )

    kas_sampler = Sampler(**sampler_args)

    assembler = kas_sampler.create_assembler()
    assembled = conv2d(assembler)
    path = assembled.convert_to_path(kas_sampler)
    print(path)

    node = kas_sampler.visit(path)
    kernelPacks, total_flops = kas_sampler.realize(
        _model.create_instance(), node, "test_manual_conv")
    print(f"Total FLOPS: {total_flops}")


def main():
    args = arg_parse_logger()
    os.makedirs(args.output_path, exist_ok=True)
    create_sample(args)


if __name__ == '__main__':
    main()
