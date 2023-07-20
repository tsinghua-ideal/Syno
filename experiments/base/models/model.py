import KAS
import torch
import thop
import logging
from torch import nn
from typing import List, Tuple
from KAS.Placeholder import Placeholder

from .placeholder import LinearPlaceholder, ConvPlaceholder


class KASModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def load_kernel(self, sampler: KAS.Sampler, node: KAS.Node, name: str=None, compile=False):
        kernel = sampler.realize(self, node, name)
        kernel_packs = kernel.construct_kernel_packs()
        placeholders = sampler._extract_placeholders(self)
        assert len(placeholders) == kernel.get_count_placeholders(), f'Kernel {kernel} has {kernel.get_count_placeholders()} placeholders, but {len(placeholders)} placeholders are found in the model'
        flops = []
        for i, (placeholder, kernel_pack) in enumerate(zip(placeholders, kernel_packs)):
            placeholder.reload(kernel_pack, compile)
            placeholder.refered_layer = None
            placeholder.set_flops(kernel.get_flops(i))
            placeholder.set_params(sum(weight.numel() for weight in kernel_pack.weights) if hasattr(kernel_pack, 'weights') else 0)
            flops.append(kernel.get_flops(i))
        assert kernel.get_total_flops() == sum(flops), f'Kernel {kernel} has {kernel.get_total_flops()} flops, but {sum(flops)} flops are found in the model'

    def profile(self, batch_size=1) -> Tuple[int, int]:
        # Get statistics (count with batch size = 1)    
        def count_placeholder(m: Placeholder, x, y):
            if m.kernel:
                m.total_ops += torch.DoubleTensor([m.flops])
            else:
                m.total_ops += m.refered_layer.total_ops

        sample_input = torch.randn((batch_size, *self.sample_input_shape())).cuda()
        flops, params = thop.profile(self, inputs=(sample_input, ), verbose=False, report_missing=False, custom_ops={
            Placeholder: count_placeholder,
            LinearPlaceholder: count_placeholder,
            ConvPlaceholder: count_placeholder
        })
        flops = flops // batch_size
        return int(flops), int(params)

    @staticmethod
    def sample_input_shape():
        assert False, 'Not implemented'

    @staticmethod
    def sampler_parameters():
        assert False, 'Not implemented'

    def initialize_weights(self):
        self.apply(KAS.init_weights)

    def forward(self, x):
        return x
