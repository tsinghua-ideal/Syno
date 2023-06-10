from torch import nn, Tensor, Size
from thop import profile
from typing import List, Callable

from KAS import Placeholder, KernelPack, Sampler
from .models import KASModule, mapping_func_conv, mapping_func_gray_conv, mapping_func_linear


class ConvNet(KASModule):
    """CNN architecture follows the implementation of DeepOBS (https://github.com/fsschneider/DeepOBS/blob/master/deepobs/tensorflow/testproblems/cifar10_3c3d.py)"""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 96, kernel_size=3, padding='same'),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(96, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )
        self.dense = nn.Sequential(
            nn.Linear(4*4*128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, image):
        B = image.size(0)
        image = image.view(B, 3, 32, 32)
        feats = self.conv(image)
        return self.dense(feats.view(B, -1))


class KASConv(KASModule):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            Placeholder(
                refered_layer=nn.Conv2d(3, 64, kernel_size=3, padding='same'),
                mapping_func=mapping_func_conv
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            Placeholder(
                refered_layer=nn.Conv2d(64, 96, kernel_size=3, padding='same'),
                mapping_func=mapping_func_conv
            ),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            Placeholder(
                refered_layer=nn.Conv2d(
                    96, 128, kernel_size=3, padding='same'),
                mapping_func=mapping_func_conv
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )
        self.dense = nn.Sequential(
            nn.Linear(4*4*128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, image: Tensor) -> Tensor:
        B = image.size(0)
        image = image.view(B, 3, 32, 32)
        feats = self.conv(image)
        return self.dense(feats.view(B, -1))


class KASGrayConv(KASModule):
    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            Placeholder(
                refered_layer=nn.Conv2d(1, 32, 5, padding='same'),
                mapping_func=mapping_func_gray_conv
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
        )
        self.linear = nn.Linear(32*7*7, 10)

    def forward(self, image: Tensor) -> Tensor:
        B = image.size(0)
        x = self.conv(image.squeeze(1))
        return self.linear(x.view(B, -1))


class KASDense(KASModule):
    def __init__(self) -> None:
        super().__init__()

        self.layers = nn.Sequential([
            Placeholder(
                refered_layer=nn.Linear(28 * 28, 1000),
                mapping_func=mapping_func_linear
            ),
            nn.ReLU(),
            Placeholder(
                refered_layer=nn.Linear(1000, 500),
                mapping_func=mapping_func_linear
            ),
            nn.ReLU(),
            Placeholder(
                refered_layer=nn.Linear(500, 100),
                mapping_func=mapping_func_linear
            ),
            nn.ReLU(),
            Placeholder(
                refered_layer=nn.Linear(100, 10),
                mapping_func=mapping_func_linear
            )
        ])

    def forward(self, image: Tensor) -> Tensor:
        B = image.size(0)
        image = image.view(B, -1)
        x = self.layers(x)
        return x
