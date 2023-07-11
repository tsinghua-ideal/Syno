from torch import nn


def get_loss_func(args):
    return nn.CrossEntropyLoss().cuda()
