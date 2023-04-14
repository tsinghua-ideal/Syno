from typing import Any, Optional, Callable, Union, Type, List

from torch import Tensor, nn
from torchvision.models.resnet import _resnet, ResNet, conv1x1, conv3x3
from timm.models.registry import register_model

from KAS import Placeholder


class KASBottleneck(nn.Module):
    """BottleneckBlock designed for KAS, the reference implementation should be conv + BatchNorm. """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # (N, C0, H, W) -> (N, C1, H, W)
        self.block1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.block2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.block3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self._bootstrap = False

    def replacePlaceHolder(self, x):
        assert(not self._bootstrap, "the module has been replaced")

        # record the shape
        N, C1, H1, W1 = x.shape
        out = self.block1(x)
        _, C2, H2, W2 = out.shape
        out = self.block2(out)
        _, C3, H3, W3 = out.shape
        out = self.block3(out)
        _, C4, H4, W4 = out.shape

        self.block1 = Placeholder({
            "N": N, "C1": C1, "H1": H1, "W1": W1, "C2": C2, "H2": H2, "W2": W2, 
        })
        self.block2 = Placeholder({
            "N": N, "C1": C2, "H1": H2, "W1": W2, "C2": C3, "H2": H3, "W2": W3,
        })
        self.block3 = Placeholder({
            "N": N, "C1": C3, "H1": H3, "W1": W3, "C2": C4, "H2": H4, "W2": W4,
        })

    def forward(self, x: Tensor) -> Tensor:
        if not self._bootstrap:
            self.replacePlaceHolder(x)
            
        identity = x.clone()

        out = self.block1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.block2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.block3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def resnet_kas(*, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet from TorchVision. Adapted to KAS format. """
    return _resnet(KASBottleneck, [2, 2, 2, 2], None, progress, **kwargs)


@register_model
# noinspection PyUnusedLocal
def kas_resnet(pretrained=False, pretrained_cfg=None, **kwargs):
    # noinspection PyTypeChecker
    model = resnet_kas()
    return model
