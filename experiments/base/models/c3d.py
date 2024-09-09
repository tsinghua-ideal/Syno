import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from KAS import init_weights

from .model import KASModel
from .placeholder import Conv3dPlaceholder

class C3D(KASModel):
    """
    The C3D network.
    """

    def __init__(self, num_classes, input_size: Tuple[int, int, int, int] = (3, 16, 112, 112)):
        super(C3D, self).__init__()

        self.input_size = input_size
        C, T, H, W = input_size
        assert T % 16 == 0
        assert H % 16 == 0 and (H // 16 + 1) % 2 == 0
        assert W % 16 == 0 and (W // 16 + 1) % 2 == 0

        self.conv1 = Conv3dPlaceholder(C, 64, kernel_size=(3, 3, 3))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) # 64, 16, 56, 56

        self.conv2 = Conv3dPlaceholder(64, 128, kernel_size=(3, 3, 3))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = Conv3dPlaceholder(128, 256, kernel_size=(3, 3, 3))
        self.conv3b = Conv3dPlaceholder(256, 256, kernel_size=(3, 3, 3))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = Conv3dPlaceholder(256, 512, kernel_size=(3, 3, 3))
        self.conv4b = Conv3dPlaceholder(512, 512, kernel_size=(3, 3, 3))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)) # 512, 2, 7, 7

        self.conv5a = Conv3dPlaceholder(512, 512, kernel_size=(3, 3, 3))
        self.conv5b = Conv3dPlaceholder(512, 512, kernel_size=(3, 3, 3))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)) # 512, 1, 4, 4

        self.size_before_flatten = T * (H + 16) * (W + 16) // 32

        self.fc6 = nn.Linear(self.size_before_flatten, 2048)
        self.fc7 = nn.Linear(2048, 2048)
        self.fc8 = nn.Linear(2048, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.flatten(1)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)

        return logits

    def sample_input_shape(self, seq_len=None):
        return self.input_size

    def sampler_parameters(self, seq_len=None):
        return {
            "input_shape": "[N, C_in: unordered, T, H, H]",
            "output_shape": "[N, C_out: unordered, T, H, H]",
            "primary_specs": ["N: 0", "C_in: 3", "C_out: 4", "T: 0", "H: 0"],
            "coefficient_specs": ["k_1=3: 2", "k_2=7: 2", "s=2: 2", "g=32: 3"],
            "fixed_io_pairs": [(0, 0)],
        }

if __name__ == '__main__':
    inputs = torch.rand(1, 3, 16, 112, 112)
    net = C3D(num_classes=51)

    outputs = net.forward(inputs)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))