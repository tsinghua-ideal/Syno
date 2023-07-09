import KAS
from torch import nn
from typing import List


class KASModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

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
