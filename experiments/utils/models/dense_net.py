from torch import nn
from KAS import Placeholder

from .model import KASModel
from .placeholder import DensePlaceholder


class FCNet(KASModel):
    def __init__(self) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 10, bias=False),
        )

    @staticmethod
    def sample_input_shape():
        return (1, 28, 28)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


class KASFCNet(KASModel):
    def __init__(self) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            DensePlaceholder(28 * 28, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    @staticmethod
    def sample_input_shape():
        return (1, 28, 28)
    
    @staticmethod
    def sampler_parameters():
        return {
            'input_shape': '[N, C_in]',
            'output_shape': '[N, C_out]',
            'primary_specs': ['N: 0', 'C_in: 2', 'C_out: 1'],
            'coefficient_specs': ['k=3: 2'],
            'fixed_io_pairs': [(0, 0)],
        }

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
