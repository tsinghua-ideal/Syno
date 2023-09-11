import torch
from torch import nn
from typing import Dict

from .KernelPack import KernelPack


class Placeholder(nn.Module):
    def __init__(self, mappings: Dict = None, refered_layer: nn.Module = None, mapping_func=None) -> None:
        super(Placeholder, self).__init__()
        assert mappings is not None or refered_layer is not None
        self.mappings = mappings
        self.refered_layer = refered_layer
        self.mapping_func = mapping_func
        self.kernel = None
        self.flops = 0
        self.params = 0
        self.build_mapping_mode = False
        self.filtered_flag = False

    def reload(self, kernel: KernelPack, compile=False) -> None:
        self.kernel = torch.compile(kernel, backend='inductor', dynamic=False, fullgraph=False) if compile else kernel

    def set_flops(self, flops: int) -> None:
        self.flops = flops
        
    def set_params(self, params: int) -> None:
        self.params = params
    
    @staticmethod
    def exclusion_condition(in_size, out_size) -> bool:
        return False

    def forward(self, x) -> torch.Tensor:
        x_size = x.size()
        out = self.refered_layer(x) if self.kernel is None else self.kernel(x)
        
        if self.build_mapping_mode:
            assert self.mapping_func is not None
            self.filtered_flag = self.exclusion_condition(x_size, out.size())
            self.mappings = self.mapping_func(x_size, out.size())
        
        return out

def remove_unsatisfied_placeholders(net: nn.Module):
    count = 0
    for name, child in net.named_children():
        if isinstance(child, Placeholder):
            if child.filtered_flag:
                count += 1
                setattr(net, name, child.refered_layer)
        elif len(list(child.named_children())) > 0:
            count += remove_unsatisfied_placeholders(child)
    return count

def build_placeholder_mappings(net: nn.Module, sample_input: torch.Tensor):
    def set_mode(v):
        for m in net.modules():
            if isinstance(m, Placeholder):
                m.build_mapping_mode = v
    
    set_mode(True)
    net(sample_input)
    set_mode(False)
