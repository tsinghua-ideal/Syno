from torch import nn
from KAS import Placeholder

from .model import KASModel


class ConvNet(KASModel):
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
            nn.Linear(4 * 4 * 128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    @staticmethod
    def sample_input_shape():
        return (3, 32, 32)

    def forward(self, image):
        batch_size = image.size(0)
        feats = self.conv(image)
        return self.dense(feats.view(batch_size, -1))


class KASConvNet(KASModel):
    def __init__(self):
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

    def sample_input_shape(self):
        return (3, 32, 32)

    def forward(self, image):
        batch_size = image.size(0)
        feats = self.conv(image)
        return self.dense(feats.view(batch_size, -1))


class KASGrayConvNet(KASModel):
    def __init__(self):
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
        self.dense = nn.Linear(32 * 7 * 7, 10)

    def sample_input_shape(self):
        return (1, 28, 28)

    def forward(self, image):
        batch_size = image.size(0)
        x = self.conv(image.squeeze(1))
        return self.dense(x.view(batch_size, -1))
