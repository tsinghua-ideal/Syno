import logging
import itertools

from utils.data import get_dataloader
from utils.models import KASConv, ModelBackup
from utils.parser import arg_parse
from utils import device
from mp_utils import Handler_client, evaluate

import torch

# Systems
import os
import sys
import logging
import traceback

# KAS
from KAS import Sampler, TreePath
from KAS.Bindings import CodeGenOptions

if __name__ == '__main__':
    # Get arguments.
    args = arg_parse()
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f'Program arguments: {args}')

    # Check available devices and set distributed.
    logging.info(f'Initializing devices ...')
    device.initialize(args)
    assert not args.distributed, 'Selector mode does not support distributed training'

    train_data_loader, validation_data_loader = get_dataloader(args)

    web_handler = Handler_client(args)
    sampler_args, train_args, extra_args = web_handler.get_args()

    sampler_args['autoscheduler'] = getattr(
        CodeGenOptions.AutoScheduler, sampler_args['autoscheduler'])

    _model = ModelBackup(KASConv, torch.randn(
        extra_args["sample_input_shape"]), extra_args["device"])
    kas_sampler = Sampler(net=_model.create_instance(), **sampler_args)

    round_range = range(
        args.kas_rounds) if args.kas_rounds > 0 else itertools.count()
    for i in round_range:
        # Sample a new kernel.
        logging.info('Requesting a new kernel ...')
        path_serial = TreePath([])
        try:
            path_serial = web_handler.get_path()
            if not path_serial:
                logging.info("fetched an empty path, shutting down. ")
                break
            elif path_serial == "ENDTOKEN":
                logging.info("fetched ENDTOKEN. Stopped......")
                # TODO: better way to end worker.
                print(
                    "Search ended. Stucking to avoid resetup. Press Ctrl-C twice to end. ")
                while True:
                    pass
            path = TreePath.deserialize(path_serial)
            logging.info(f"Received {path}")
            try:
                state, reward = evaluate(
                    path, train_data_loader, validation_data_loader, _model, kas_sampler, train_args, extra_args)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    break
                state = "FAILURE_EVALUATION"
                traceback.print_exc()
            if state == "SUCCESS":
                web_handler.success(path_serial, state, reward)
            else:
                web_handler.failure(path_serial, state)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                break
            state = "FAILURE_DEATH"
            try:
                web_handler.failure(path_serial, state)
            except:
                pass
            traceback.print_exc()
