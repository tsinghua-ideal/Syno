import torch
from http.server import HTTPServer

# Systems
import time
import random
import os
import sys
import logging
import traceback
import json
from copy import deepcopy

# KAS
from KAS import Sampler
from KAS.Bindings import CodeGenOptions

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from utils.models import KASConv, ModelBackup
from utils.parser import arg_parse

from mp_utils import Handler_server, MCTSTrainer

if __name__ == '__main__':

    # set logging level
    logging.getLogger().setLevel(logging.INFO)

    args = arg_parse()
    print(args)
    use_cuda = torch.cuda.is_available()

    os.makedirs(args.kas_sampler_save_dir, exist_ok=True)

    training_params = dict(
        lr=0.001,
        momentum=0.9,
        epochs=50,
        val_period=5,
        use_cuda=use_cuda
    )
    sampler_params = dict(
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
        max_flops=1176494080,  # manual conv size
        save_path=args.kas_sampler_save_dir,
        cuda=use_cuda,
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
        sample_input_shape=(args.batch_size, *args.input_size),
        device="cuda" if use_cuda else "cpu"
    )

    arguments = deepcopy(dict(
        sampler_args=sampler_params,
        train_args=training_params,
        extra_args=extra_args
    ))
    arguments['sampler_args']['autoscheduler'] = str(
        arguments['sampler_args']['autoscheduler'])[14:]  # HACK: serialize the enum

    _model = ModelBackup(KASConv, torch.randn(
        extra_args["sample_input_shape"]), "cpu")
    kas_sampler = Sampler(net=_model.create_instance(), **sampler_params)

    searcher = MCTSTrainer(
        kas_sampler,
        arguments,
        mcts_iterations=args.kas_iterations,
        leaf_parallelization_number=args.kas_leaf_parallelization_number,
        simulate_retry_limit=args.kas_simulate_retry_limit,
        virtual_loss_constant=args.kas_tree_parallelization_virtual_loss_constant
    )

    class MCTSHandler(Handler_server):
        def __init__(self, *args, **kwargs) -> None:
            self.mcts = searcher
            super().__init__(*args, **kwargs)

    # Start HTTP server.
    print(f'Starting listening at {args.host}:{args.port}')
    server = HTTPServer((args.host, args.port), MCTSHandler)
    server.serve_forever()
