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

from train import train

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from utils.models import KASGrayConv as KASConv, ModelBackup


def evaluate(path: TreePath, _model: ModelBackup, kas_sampler: Sampler, train_args: dict, extra_args: dict) -> Tuple[str, float]:
    """
    Evaluate a serialized path. 
    Steps: deserialize -> realize -> train -> evaluate

    extra_args:
        - max_macs
        - min_macs
        - max_model_size
        - min_model_size
        - prefix
        - model_type: str, KASConv for example
        - sample_input_shape: Tuple[int]
        - device: str, cuda:i or cpu

    Return
    ----------
    State, Accuracy
    """

    node = kas_sampler.visit(path)
    kernelPacks, total_flops = kas_sampler.realize(
        model, node, extra_args["prefix"])
    model = _model.restore_model_params(model, kernelPacks)

    model_macs = _model.base_macs + total_flops * 2
    logging.info("Model macs: {}".format(model_macs))
    if model_macs > extra_args['max_macs']:
        logging.info(
            f"Model macs {model_macs} exceeds {extra_args['max_macs']}, skipping")
        return "FAILED because of too big model macs.", 0
    elif model_macs < extra_args['min_macs']:
        logging.info(
            f"Model macs {model_macs} below {extra_args['min_macs']}, skipping")
        return "FAILED because of too small model macs.", 0

    model_size = sum([p.numel() for p in model.parameters()])
    logging.info("Model size: {}".format(model_size))
    if model_size > extra_args['max_model_size']:
        logging.info(
            f"Model size {model_size} exceeds {extra_args['max_model_size']}, skipping")
        return "FAILED because of too big model size.", 0
    elif model_size < extra_args['min_model_size']:
        logging.info(
            f"Model size {model_size} below {extra_args['min_model_size']}, skipping")
        return "FAILED because of too small model size.", 0

    _, val_error, _ = train(model, **train_args, verbose=True)
    accuracy = 1. - min(val_error)
    assert 0 <= accuracy <= 1
    return "SUCCESS", accuracy
