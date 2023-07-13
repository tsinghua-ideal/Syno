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

    def profile(self, batch_size=1) -> Tuple[int, int]:
        # Get statistics (count with batch size = 1)    
        def count_placeholder(m: Placeholder, x, y):
            if m.kernel:
                m.total_ops += torch.DoubleTensor([int(m.flops)])
                # TODO: count params
                # m.total_params += torch.DoubleTensor([int(m.params)])
            # TODO: count refered layer

        sample_input = torch.randn((batch_size, *self.sample_input_shape())).cuda()
        flops, params = thop.profile(self, inputs=(sample_input, ), verbose=False, report_missing=False, custom_ops={
            Placeholder: count_placeholder,
            LinearPlaceholder: count_placeholder,
            ConvPlaceholder: count_placeholder
        })
        flops = flops // batch_size
        return flops, params

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
