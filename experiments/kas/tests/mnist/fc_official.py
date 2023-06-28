import torch

# Systems
import time
import os
import sys
import logging

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from train import train
from utils.data import get_dataloader
from utils.models import FCNet, ModelBackup
from utils.parser import arg_parse
from utils.config import parameters

if __name__ == '__main__':

    # set logging level
    logging.getLogger().setLevel(logging.INFO)

    args = arg_parse()
    assert args.dataset == 'mnist'
    print(args)

    training_params, sampler_params, extra_args = parameters(args)
    train_data_loader, validation_data_loader = get_dataloader(args)
    model_ = ModelBackup(FCNet, torch.randn(
        extra_args["sample_input_shape"]), "cpu")
    model = model_.create_instance()

    start = time.time()
    train_error, val_error, _ = train(
        model, train_data_loader, validation_data_loader, args, **training_params, verbose=True)
    accuracy = 1. - min(val_error)

    print("Test Complete, elapsed {} seconds, accuracy {}. ".format(
        time.time() - start, accuracy))
