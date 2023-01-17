import torch
from torch import nn

from .KernelPack import KernelPack


class Placeholder(nn.Module):
    def __init__(self, mappings: dict[str, int]):
        super(Placeholder, self).__init__()
        self.mappings = mappings
        self.kernel = None

    def reload(self, kernel: KernelPack):
        self.kernel = kernel

    def forward(self, x):
        return self.kernel(x)
