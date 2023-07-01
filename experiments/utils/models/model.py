from torch import nn
from typing import List


class KASModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    @staticmethod
    def sample_input_shape():
        assert False, 'Not implemented'

    @staticmethod
    def sampler_parameters():
        assert False, 'Not implemented'

    def _initialize_weight(self):
        self.apply(self.init_weights)

    def forward(self, x):
        return x
