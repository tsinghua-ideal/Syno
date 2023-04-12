from typing import Any

from torch import Tensor, nn
from torchvision.models.resnet import _resnet, ResNet
from timm.models.registry import register_model

from KAS import Placeholder

class KASBottleneckBlock(nn.Module):
    """BottleneckBlock designed for KAS, the reference implementation should be conv + BatchNorm. """
    def __init__(self) -> None:
        super().__init__()
        self.block1 = Placeholder()  # (N, C0, H, W) -> (N, C1, H, W)
        self.block2 = Placeholder()  # (N, C1, H, W) -> (N, C2, H, W)
        self.block3 = Placeholder()  # (N, C1, H, W) -> (N, C2, H, W)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        identity = x.clone()

        out = self.block1(x)
        out = self.relu(out)

        out = self.block2(out)
        out = self.relu(out)

        out = self.block3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def resnet_kas(*, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet from TorchVision. Adapted to KAS format. """
    return _resnet(KASBottleneckBlock, [2, 2, 2, 2], None, progress, **kwargs)

@register_model
# noinspection PyUnusedLocal
def kas_resnet(pretrained=False, pretrained_cfg=None, **kwargs):
    # noinspection PyTypeChecker
    model = resnet_kas()
    return model
