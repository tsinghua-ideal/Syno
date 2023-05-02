from torch import nn, Tensor

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


class KASConv(nn.Module):
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

        self.layers = None

    def bootstraped(self) -> bool:
        return self.layers is not None

    def generatePlaceHolder(self, x: Tensor) -> Tensor:
        """Replace blocks with Placeholder for search"""
        assert not self.bootstraped(), "the module has been replaced"

        layers = []

        # record the shape
        for block in self.blocks:

            if isinstance(block, nn.Conv2d):
                N, C1, H, W = x.size()

            x = block(x)

            if isinstance(block, nn.Conv2d):
                N2, C2, H2, W2 = x.size()
                assert N2 == N, f"Batch size change detected! {N} -> {N2}. "
                assert H2 == H and W2 == W, "Not using same padding. "
                block = Placeholder(
                    {"N": N, "C_in": C1, "C_out": C2, "H": H, "W": W})

            layers.append(block)

        self.layers = nn.Sequential(*layers)

        return x

    def forward(self, image: Tensor) -> Tensor:
        B = image.size(0)
        image = image.view(B, 1, 28, 28)

        x = self.layers(image)

        return self.linear(x.view(B, -1))


class KASGrayConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([
            nn.Conv2d(1, 32, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
        ])
        self.linear = nn.Linear(32*7*7, 10)

        self.layers = None

    def bootstraped(self) -> bool:
        return self.layers is not None

    def generatePlaceHolder(self, x: Tensor) -> Tensor:
        """Replace blocks with Placeholder for search"""
        assert not self.bootstraped(), "the module has been replaced"

        layers = []

        # record the shape
        for block in self.blocks:

            if isinstance(block, nn.Conv2d):
                N, C1, H, W = x.size()
                assert C1 == 1

            x = block(x)

            if isinstance(block, nn.Conv2d):
                N2, C2, H2, W2 = x.size()
                assert N2 == N, f"Batch size change detected! {N} -> {N2}. "
                assert H2 == H and W2 == W, "Not using same padding. "
                block = Placeholder(
                    {"N": N, "C_out": C2, "H": H, "W": W})

            layers.append(block)

        self.layers = nn.ModuleList(layers)

        return x

    def _initialize_weight(self):
        self.linear.weight.data.normal_(0, 0.01)
        self.linear.bias.data.zero_()

    def forward(self, image: Tensor) -> Tensor:
        B = image.size(0)
        x = image.squeeze(1)

        for layer in self.layers:
            x = layer(x)

        return self.linear(x.view(B, -1))
