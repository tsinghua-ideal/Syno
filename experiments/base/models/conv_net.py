import torch
from torch import nn

from .model import KASModel
from .placeholder import ConvPlaceholder


class ConvNet(KASModel):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ConvPlaceholder(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            ConvPlaceholder(64, 96, 3),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            ConvPlaceholder(96, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    @staticmethod
    def sample_input_shape():
        return (3, 32, 32)
    
    @staticmethod
    def sampler_parameters():
         return {
            'input_shape': '[N, C_in, H, W]',
            'output_shape': '[N, C_out, H, W]',
            'primary_specs': ['N: 0', 'C_in: 6', 'C_out: 6', 'H: 6', 'W: 6'],
            'coefficient_specs': ['k=3: 8', 's=2: 8'],
            'fixed_io_pairs': [(0, 0)],
        }    

    def forward(self, image):
        batch_size = image.size(0)
        feats = self.conv(image)
        return self.fc(feats.reshape(batch_size, -1))
    

class Residual(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()

        self.act = nn.GELU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(ic, oc, 3, padding='same')
        self.norm1 = nn.BatchNorm2d(oc)
        self.conv2 = nn.Conv2d(oc, oc, 3, padding='same')
        self.norm2 = nn.BatchNorm2d(oc)

    def forward(self, x):
        x = self.act(self.norm1(self.pool1(self.conv1(x))))
        return self.act(x + self.norm2(self.conv2(x)))


class SpeedyResNet(KASModel):
    def __init__(self):
        super().__init__()
        self.whiten = nn.Conv2d(3, 12, 2, padding=0)
        self.project = nn.Conv2d(12, 32, 1)
        self.act = nn.GELU()
        self.residual1 = Residual(32, 64)
        self.residual2 = Residual(64, 256)
        self.residual3 = Residual(256, 512)
        self.linear = nn.Linear(512, 10)

    @staticmethod
    def sample_input_shape():
        return (3, 32, 32)

    def forward(self, x):
        x = self.act(self.project(self.whiten(x)))
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        x = torch.amax(x, dim=(2, 3))
        return self.linear(x)
