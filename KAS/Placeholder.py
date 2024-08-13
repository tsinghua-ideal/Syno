import torch
from torch import nn
from enum import Enum
from typing import Dict, Optional, Union

from .KernelPack import KernelPack


class ExportType(Enum):
    ONNX = 'onnx'
    RELAX = 'relax'

class ONNXKernelMark(torch.autograd.Function):
    @staticmethod
    def symbolic(g: torch._C.Graph, x: torch._C.Node, kernel_id: int):
        return g.op('kas::ONNXKernelMark', x, kernel_id_i=kernel_id).setType(x.type())

    @staticmethod
    def forward(ctx, x: torch.Tensor, kernel_id):
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


class Placeholder(nn.Module):
    def __init__(self, mappings: Dict = None, referred_layer: nn.Module = None, mapping_func=None, group_identifier=None) -> None:
        super(Placeholder, self).__init__()
        assert mappings is not None or referred_layer is not None
        self.mappings = mappings
        self.referred_layer = referred_layer
        self.mapping_func = mapping_func
        self.kernel = None
        self.flops = 0
        self.params = 0
        self.build_mapping_mode = False
        self.filtered_flag = False
        # For model export.
        self.export_type: Optional[ExportType] = None
        self.export_id: Optional[int] = None
        self.group = group_identifier

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
        if self.export_type is not None:
            assert self.referred_layer is not None
            assert self.export_id is not None
            if self.export_type == ExportType.ONNX:
                return ONNXKernelMark.apply(self.referred_layer(x), self.export_id)
            elif self.export_type == ExportType.RELAX:
                # This is a hack. There is no way to pass information to Relax, so we do this and run a transformation pass in Relax.
                return torch.exp(self.referred_layer(x - self.export_id))
            else:
                assert False, f'Unknown export type {self.export_type}'

        x_size = x.size()
        out = self.referred_layer(x) if self.kernel is None else self.kernel(x)
        
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
                setattr(net, name, child.referred_layer)
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

def enable_export_for_placeholders(net: nn.Module, export_type: ExportType, enabled: bool = True):
    export_id = 0
    def set_mode(module: nn.Module):
        nonlocal export_id
        if isinstance(module, Placeholder):
            if enabled:
                module.export_type = export_type
                module.export_id = export_id
                export_id += 1
            else:
                module.export_type = None
                module.export_id = None
    for child in net.modules():
        set_mode(child)
