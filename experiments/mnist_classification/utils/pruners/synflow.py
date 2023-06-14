import torch
from torch import nn
from torch.utils.data import DataLoader

from typing import List

# KAS
from KAS import Placeholder


def get_batch(dataloader):
    images, labels = next(iter(dataloader))
    return images, labels


def syn_flow(net: nn.Module, input_shape: List[int], divide_count: bool = True,
             device: str = 'cuda', threshold: float = 0, only_kernel: bool = False):
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

    input_shape[0] = 1

    # Linearize
    signs = linearize(net)
    net.zero_grad()
    net = net.double().to(device)
    images = torch.ones(input_shape).double().to(device)
    out = net(images)
    torch.sum(out).backward()

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
    if torch.isnan(syn_flow):
        return 0
    syn_flow = syn_flow / count if divide_count else syn_flow
    if threshold == 0:
        return syn_flow
    return 0 if syn_flow > threshold else syn_flow
