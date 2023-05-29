from torch.utils.data import DataLoader

# Systems
from typing import Tuple
import os
import sys
import logging

# KAS
from KAS import Sampler, TreePath

from train import train

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from utils.models import ModelBackup


def evaluate(path: TreePath, train_loader: DataLoader, val_loader: DataLoader, _model: ModelBackup, kas_sampler: Sampler, train_args: dict, extra_args: dict) -> Tuple[str, float]:
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

    logging.info(f"Evaluation started")
    node = kas_sampler.visit(path)
    logging.info(f"Node generated. ")

    estimate_model_macs = _model.base_macs + \
        node.estimate_total_flops_as_final() * 2
    logging.info("(Estimated) Model macs: {}".format(estimate_model_macs))
    if estimate_model_macs > extra_args['max_macs']:
        logging.info(
            f"(Estimated) Model macs {estimate_model_macs} exceeds {extra_args['max_macs']}, skipping")
        return "FAILED_exceeding_model_macs.", 0
    elif estimate_model_macs < extra_args['min_macs']:
        logging.info(
            f"(Estimated) Model macs {estimate_model_macs} below {extra_args['min_macs']}, skipping")
        return "FAILED_small_model_macs.", 0

    model = _model.create_instance()
    kernelPacks, total_flops = kas_sampler.realize(
        model, node, extra_args["prefix"])
    logging.info(f"kernelPack generated. ")
    model = _model.restore_model_params(model, kernelPacks)
    logging.info(f"Model constructed. ")

    model_macs = _model.base_macs + total_flops * 2
    logging.info("Model macs: {}".format(model_macs))
    if model_macs > extra_args['max_macs']:
        logging.info(
            f"Model macs {model_macs} exceeds {extra_args['max_macs']}, skipping")
        return "FAILED_exceeding_model_macs.", 0
    elif model_macs < extra_args['min_macs']:
        logging.info(
            f"Model macs {model_macs} below {extra_args['min_macs']}, skipping")
        return "FAILED_small_model_macs.", 0

    model_size = sum([p.numel() for p in model.parameters()])
    logging.info("Model size: {}".format(model_size))
    if model_size > extra_args['max_model_size']:
        logging.info(
            f"Model size {model_size} exceeds {extra_args['max_model_size']}, skipping")
        return "FAILED_exceeding_model_size.", 0
    elif model_size < extra_args['min_model_size']:
        logging.info(
            f"Model size {model_size} below {extra_args['min_model_size']}, skipping")
        return "FAILED_small_model_size.", 0

    _, val_error, _ = train(model, train_loader,
                            val_loader, **train_args, verbose=True)
    accuracy = 1. - min(val_error)
    assert 0 <= accuracy <= 1
    return "SUCCESS", accuracy
