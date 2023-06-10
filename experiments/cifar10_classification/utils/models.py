from torch import nn, Tensor, Size
from thop import profile
from typing import List, Callable

from KAS import Placeholder, KernelPack, Sampler


class ModelBackup:
    def __init__(self, model: Callable, sample_input: Tensor, device='cuda:0') -> None:
        self._model_builder = model
        self._sample_input = sample_input.to(device)

        self._model = self._model_builder().to(device)
        macs, params = profile(self._model, inputs=(
            self._sample_input, ), verbose=False)
        print(
            f"Referenced model has {round(macs / 1e9, 3)}G MACs and {round(params / 1e6, 3)}M parameters ")
        self.base_macs = macs
        print(f"Base MACs is {self.base_macs}")

    def create_instance(self) -> nn.Module:
        self._model._initialize_weight()
        return self._model

    def restore_model_params(self, model, pack: List[KernelPack]):
        """
        Restore model parameters and replace the selected parameters with pack.
        """
        assert len(pack) > 0, "Not detected any placeholders! "
        assert isinstance(pack[0], KernelPack
                          ), f"elements in pack are not valid! {type(pack[0])}"
        Sampler.replace(model, pack)
        return model


class ConvNet(nn.Module):
    """CNN architecture follows the implementation of DeepOBS (https://github.com/fsschneider/DeepOBS/blob/master/deepobs/tensorflow/testproblems/cifar10_3c3d.py)"""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding='valid'),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 96, kernel_size=3, padding='valid'),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.dense = nn.Sequential(
            nn.Linear(3*3*128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, image):
        B = image.size(0)
        image = image.view(B, 3, 32, 32)
        feats = self.conv(image)
        return self.dense(feats.view(B, -1))


def mapping_func_conv(in_size: Size, out_size: Size):
    N, C1, H, W = in_size
    N2, C2, H2, W2 = out_size
    assert N2 == N, f"Batch size change detected! {N} -> {N2}. "
    assert H2 == H and W2 == W, f"Not using same padding. {in_size}->{out_size}"
    mapping = {"N": N, "C_in": C1, "C_out": C2, "H": H, "W": W}
    return mapping


def mapping_func_linear(in_size: Size, out_size: Size):
    N, C1 = in_size
    N2, C2 = out_size
    assert N2 == N, f"Batch size change detected! {N} -> {N2}. "
    mapping = {"N": N, "C_in": C1, "C_out": C2}
    return mapping


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

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def _initialize_weight(self):
        self.apply(self.init_weights)

    def forward(self, image: Tensor) -> Tensor:
        return image


class KASConv(KASModule):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 96, kernel_size=5, padding='same'),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(96, 128, kernel_size=3, padding='same'),
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
        x = image
        assert x.shape == (B, 3, 32, 32)

        if self.bootstraped():
            for layer in self.layers:
                # print("Passing", layer, x.size())
                x = layer(x)
        else:
            for block in self.blocks:
                x = block(x)
        return self.dense(x.view(B, -1))


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


class KASDense(KASModule):
    def __init__(self) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([
            nn.Linear(28 * 28, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        ])

    def forward(self, image: Tensor) -> Tensor:
        B = image.size(0)
        image = image.view(B, -1)

        if self.bootstraped():
            for layer in self.layers:
                x = layer(x)
        else:
            x = image
            for block in self.blocks:
                x = block(x)

        return x
