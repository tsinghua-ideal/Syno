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
from utils.models import KASFC, ModelBackup
from utils.parser import arg_parse
from utils.config import parameters

from mp_utils import Handler_server, MCTSTrainer

if __name__ == '__main__':

    # set logging level
    logging.getLogger().setLevel(logging.DEBUG)

    args = arg_parse()
    print(args)
    use_cuda = torch.cuda.is_available()

    os.makedirs(args.kas_sampler_save_dir, exist_ok=True)

    training_params, sampler_params, extra_args = parameters(args)

    arguments = deepcopy(dict(
        sampler_args=sampler_params,
        train_args=training_params,
        extra_args=extra_args
    ))
    arguments['sampler_args']['autoscheduler'] = str(
        arguments['sampler_args']['autoscheduler'])[14:]  # HACK: serialize the enum

    _model = ModelBackup(KASFC, torch.randn(
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
