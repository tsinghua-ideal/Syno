import torch
from torch import nn
from typing import Dict

from .KernelPack import KernelPack


class Placeholder(nn.Module):
    def __init__(self, mappings: Dict[str, int] = None, refered_layer: nn.Module = None, mapping_func=None) -> None:
        super(Placeholder, self).__init__()
        assert mappings is not None or refered_layer is not None
        self.mappings = mappings
        self.refered_layer = refered_layer
        # takes in.size() and out.size() and returns the mapping
        self.mapping_func = mapping_func
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
        if self.mappings is None:
            out = self.refered_layer(x)
            self.mappings = self.mapping_func(
                x.size(), out.size())
            print("PlaceHolder initialized.")
            return out
        else:
            return self.kernel(x)
