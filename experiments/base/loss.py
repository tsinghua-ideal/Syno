from torch import nn


def get_loss_func(args):
    return nn.CrossEntropyLoss()


def get_gnn_loss_func(args):
    return nn.NLLLoss()
