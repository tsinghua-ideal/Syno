from torch import nn, Tensor
from thop import profile

from KAS import Placeholder


class ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=False),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0, 1, ceil_mode=False)
        )
        self.linear = nn.Linear(32*7*7, 10)

    def forward(self, image):
        B = image.size(0)
        image = image.view(B, 1, 28, 28)
        feats = self.conv(image)
        return self.linear(feats.view(B, -1))


class KASModule(nn.Module):
    """
    Not Fully Implemented!!!

    A wrapper of nn.Module for KAS. 

    To use it, redefine the blocks and forward functions. KASModule will automatically replace all conv layers with placeholders. 
    """

    def __init__(self) -> None:
        super().__init__()

        self.blocks = None
        self.layers = None

    def bootstraped(self) -> bool:
        return self.layers is not None

    def generatePlaceHolder(self, x: Tensor) -> Tensor:
        """Replace blocks with Placeholder for search"""
        assert not self.bootstraped(), "the module has been replaced"

        layers = []
        replaced_macs = 0

        # record the shape
        for block in self.blocks:

            if isinstance(block, nn.Conv2d):
                replaced_macs += profile(block, inputs=(x, ), verbose=False)[0]
                N, C1, H, W = x.size()

            x = block(x)

            if isinstance(block, nn.Conv2d):
                N2, C2, H2, W2 = x.size()
                assert N2 == N, f"Batch size change detected! {N} -> {N2}. "
                assert H2 == H and W2 == W, "Not using same padding. "
                block = Placeholder(
                    {"N": N, "C_out": C2, "H": H, "W": W})

            layers.append(block)

        self.layers = nn.ModuleList(layers)

        return replaced_macs

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

    def _initialize_weight(self):
        self.apply(self.init_weights)

    def forward(self, image: Tensor) -> Tensor:
        return image


class KASConv(KASModule):
    def __init__(self) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([
            nn.Conv2d(1, 16, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ])

        self.linear = nn.Linear(32*7*7, 10)

    def forward(self, image: Tensor) -> Tensor:
        B = image.size(0)
        image = image.view(B, 1, 28, 28)

        x = self.layers(image)

        return self.linear(x.view(B, -1))


class KASGrayConv(KASModule):
    def __init__(self) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([
            nn.Conv2d(1, 32, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
        ])
        self.linear = nn.Linear(32*7*7, 10)

    def forward(self, image: Tensor) -> Tensor:
        B = image.size(0)

        if self.bootstraped():
            x = image.squeeze(1)
            for layer in self.layers:
                x = layer(x)
        else:
            x = image
            for block in self.blocks:
                x = block(x)

        return self.linear(x.view(B, -1))
