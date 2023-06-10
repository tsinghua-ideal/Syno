
import torch
from torch import nn, Tensor

# Systems
import time
import random
from typing import List
import os
import sys
import logging
from thop import profile

# KAS
from KAS import Sampler, KernelPack, Assembled, Assembler
from KAS.Bindings import CodeGenOptions


if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from train import train
from utils.data import get_dataloader
from utils.models import KASConv, ModelBackup
from utils.parser import arg_parse
from utils.config import parameters

if __name__ == '__main__':
    args = arg_parse()
    print(args)
    use_cuda = torch.cuda.is_available()

    training_params, sampler_params, extra_args = parameters(args)
    train_data_loader, validation_data_loader = get_dataloader(args)

    model_ = ModelBackup(KASConv, torch.randn(
        extra_args["sample_input_shape"]), "cuda")
    model = model_.create_instance()

    start = time.time()
    train_error, val_error, _ = train(
        model, train_data_loader, validation_data_loader, args, **training_params, verbose=True)
    accuracy = 1. - min(val_error)

    print("Test Complete, elapsed {} seconds, accuracy {}. ".format(
        time.time() - start, accuracy))
