from torch import nn, Tensor

from KAS import Placeholder
from .models import KASModule, mapping_func_linear


class FCNet(KASModule):
    def __init__(self) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 10, bias=False),
        )

    def forward(self, image: Tensor) -> Tensor:
        B = image.size(0)
        x = image.view(B, -1)
        x = self.layers(x)
        return x


class KASFC(KASModule):
    def __init__(self) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Placeholder(
                refered_layer=nn.Linear(28 * 28, 64),
                mapping_func=mapping_func_linear
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            Placeholder(
                refered_layer=nn.Linear(64, 10),
                mapping_func=mapping_func_linear
            ),
        )

    def forward(self, image: Tensor) -> Tensor:
        B = image.size(0)
        x = image.view(B, -1)
        x = self.layers(x)
        return x
