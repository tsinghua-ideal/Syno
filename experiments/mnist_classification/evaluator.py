import logging
import itertools

from utils.data import get_dataloader
from utils.models import KASGrayConv as KASConv, ModelBackup
from utils.parser import arg_parse
from utils import device
from mp_utils import Handler_client, evaluate

import torch

# Systems
import os, sys
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
    extra_args["sample_input_shape"] = (
        extra_args["batch_size"], *train_data_loader.dataset[0][0].shape)

    _model = ModelBackup(KASConv, torch.randn(
        extra_args["sample_input_shape"]), extra_args["device"])
    kas_sampler = Sampler(net=_model.create_instance(), **sampler_args)

    round_range = range(
        args.kas_rounds) if args.kas_rounds > 0 else itertools.count()
    for i in round_range:
        # Sample a new kernel.
        logging.info('Requesting a new kernel ...')
        path_serial = web_handler.get_path()
        if not path_serial:
            logging.info("fetched an empty path, shutting down. ")
            break
        try:
            logging.info(f"Received {path_serial}")
            path = TreePath.deserialize(path_serial)
            try:
                state, reward = evaluate(
                    path, train_data_loader, validation_data_loader, _model, kas_sampler, train_args, extra_args)
            except:
                state = "FAILURE_EVALUATION"
                traceback.print_exc()
            if state == "SUCCESS":
                web_handler.success(path_serial, state, reward)
            else:
                web_handler.failure(path_serial, state)
        except Exception as e:
            state = "FAILURE_DEATH"
            web_handler.failure(path_serial, state)
            traceback.print_exc()
            break
