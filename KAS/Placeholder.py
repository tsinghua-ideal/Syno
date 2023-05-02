import torch
from torch import nn

from .KernelPack import KernelPack


class Placeholder(nn.Module):
    def __init__(self, mappings: dict[str, int]) -> None:
        super(Placeholder, self).__init__()
        self.mappings = mappings
        self.kernel = None
        self._flops = 0

    def reload(self, kernel: KernelPack) -> None:
        self.kernel = kernel

    def set_flops(self, flops: int) -> None:
        self._flops = flops

    @staticmethod
    def count_macs(layer, *args) -> int:
        assert layer.kernel is not None, "Kernel not loaded"
        return layer._flops * 2

    def forward(self, x) -> torch.Tensor:
        return self.kernel(x)
