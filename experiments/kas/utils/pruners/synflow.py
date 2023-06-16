import torch
from torch import nn, Tensor

from typing import List, Literal
import math

# KAS
from KAS import Placeholder

def syn_flow(net: nn.Module, input_shape: List[int], divide_count: bool = True,
             device: str = 'cuda', threshold: float = 0, only_kernel: bool = False) -> float | Literal[0]:

    net = net.to(device)

    @torch.no_grad()
    def linearize(net: nn.Module):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def non_linearize(net: nn.Module, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # Linearize
    signs = linearize(net)
    net.zero_grad()
    images = torch.ones(input_shape).to(device)
    out: Tensor = net(images)
    out.mean(0).sum().backward()

    # Calculate syn-flow (averaging by batch)
    syn_flow, count = 0, 0
    if only_kernel:
        for m in net.modules():
            if isinstance(m, Placeholder):
                for p in m.parameters():
                    if p.grad is not None:
                        syn_flow += torch.sum(torch.abs(p * p.grad)).item()
                        count += torch.numel(p)
    else:
        for p in net.parameters():
            if p.grad is not None:
                syn_flow += torch.sum(torch.abs(p * p.grad)).item()
                count += torch.numel(p)
    syn_flow /= images.size(0)

    # Restore and return
    assert count > 0
    non_linearize(net, signs)
    if math.isnan(syn_flow):
        return 0
    syn_flow = syn_flow / count if divide_count else syn_flow
    if threshold == 0:
        return syn_flow
    return 0 if syn_flow > threshold else syn_flow
