import warnings

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from .utils import get_batch


def add_hooks(net: nn.Module) -> list:
    def counting_forward_hook(module, i, o) -> None:
        if not hasattr(module, 'visited_backwards'):
            return
        if isinstance(i, tuple):
            i = i[0]
        device = i.get_device()
        i = i.view(i.size(0), -1)
        x = (i > 0).float()
        K = x @ x.t()
        K2 = (1.0 - x) @ (1.0 - x.t())
        K = K.to(device)
        K2 = K2.to(device)
        net.K = net.K + K.cpu() + K2.cpu()

    def counting_backward_hook(module, i, o) -> None:
        module.visited_backwards = True

    handles = []
    for name, module in net.named_modules():
        if isinstance(module, nn.ReLU) and 'kernel_relu' in name:
            handles.append(module.register_forward_hook(counting_forward_hook))
            handles.append(module.register_backward_hook(
                counting_backward_hook))
    return handles


def naswot(net: nn.Module, dataloader: DataLoader, device: str = 'cuda') -> Tensor:
    net = net.to(device)
    images, _ = get_batch(dataloader)
    images = images.to(device)
    batch_size = images.size()[0]
    net.K = torch.zeros((batch_size, batch_size))
    handles = add_hooks(net)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        net.zero_grad()
        y = net(torch.zeros_like(images))
        y.backward(torch.ones_like(y))
        net(images)

    for handle in handles:
        handle.remove()
    score = torch.logdet(net.K)
    del net.K
    return score
