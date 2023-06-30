from torch import nn
from timm.optim import create_optimizer_v2, optimizer_kwargs


def get_optimizer(model: nn.Module, args):
    return create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
