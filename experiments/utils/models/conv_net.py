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
            'primary_specs': ['N: 0', 'C_in: 2', 'C_out: 2', 'H: 2', 'W: 2'],
            'coefficient_specs': ['k=3: 2'],
            'fixed_io_pairs': [(0, 0)],
        }    

    def forward(self, image):
        batch_size = image.size(0)
        feats = self.conv(image)
        return self.fc(feats.reshape(batch_size, -1))
