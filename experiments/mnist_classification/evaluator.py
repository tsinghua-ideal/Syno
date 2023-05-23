import logging
import itertools

from mp_utils import Handler_client, evaluate
from utils import parser, device

from train import train
from utils.data import get_dataloader
from utils.models import KASGrayConv as KASConv, ModelBackup
from utils.parser import arg_parse

import torch
from torch import nn, Tensor

# Systems
from typing import List, Tuple
import os
import sys
import logging
from thop import profile

# KAS
from KAS import MCTS, Sampler, KernelPack, Node, Path, Placeholder, TreePath

if __name__ == '__main__':
    # Get arguments.
    args = arg_parse()
    logging.info(f'Program arguments: {args}')

    # Check available devices and set distributed.
    logging.info(f'Initializing devices ...')
    device.initialize(args)
    assert not args.distributed, 'Selector mode does not support distributed training'

    train_data_loader, validation_data_loader = get_dataloader(args)

    web_handler = Handler_client(args)
    sampler_args, train_args, extra_args = web_handler.get_args()
    _model = ModelBackup(KASConv, torch.randn(
        extra_args["sample_input_shape"]), extra_args["device"])
    kas_sampler = Sampler(net=_model.create_instance(), **sampler_args)
    extra_args["sample_input_shape"] = (
        extra_args["batch_size"], *train_data_loader.dataset[0][0].shape)

    round_range = range(
        args.kas_rounds) if args.kas_rounds > 0 else itertools.count()
    for i in round_range:
        # Sample a new kernel.
        logging.info('Requesting a new kernel ...')
        try:
            path_serial = web_handler.get_path()
            path = TreePath.deserialize(path_serial)
            state, reward = evaluate(
                path, _model, kas_sampler, train_args, extra_args)
            if state == "SUCCESS":
                web_handler.success(path_serial, reward)
            else:
                web_handler.failure(path_serial)
        except:
            break
