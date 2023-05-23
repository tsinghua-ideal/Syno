import torch
from torch import nn, Tensor

import math
from http.server import HTTPServer

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

from mp_utils import Handler_server, MCTSTrainer

if __name__ == '__main__':

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

    extra_args = dict(
        max_macs=int(args.kas_max_macs * 1e9),
        min_macs=int(args.kas_min_macs * 1e9),
        max_model_size=int(args.kas_max_params * 1e6),
        min_model_size=int(args.kas_min_params * 1e6),
        prefix="",
        model_type="KASConv",  # TODO: dynamically load the module
        batch_size=args.batch_size,
        device=torch.device("cuda" if use_cuda else "cpu")
    )

    arguments = dict(
        sampler_params=sampler_params,
        training_params=training_params,
        extra_args=extra_args
    )

    _model = ModelBackup(KASConv, torch.randn(
        extra_args["sample_input_shape"]), extra_args["device"])
    kas_sampler = Sampler(net=_model.create_instance(), **sampler_params)

    searcher = MCTSTrainer(kas_sampler, arguments,
                           mcts_iterations=args.kas_iterations, leaf_parallelization_number=args.kas_leaf_parallelization_number)

    class MCTSHandler(Handler_server):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.mcts = searcher

    # Start HTTP server.
    print(f'Starting listening at {args.host}:{args.port}')
    server = HTTPServer((args.host, args.port), MCTSHandler)
    server.serve_forever()
