import logging
from torch import nn
from torchvision import models
from typing import Dict, Optional, Tuple
import math

from .model import KASModel
from .placeholder import ConvPlaceholder


def replace_conv2d_filter(conv: nn.Conv2d, name: str) -> Optional[nn.Module]:
    # TODO: maybe relax the requirements

    def same_padding(k, p):
        return k[0] == 2 * p[0] + 1 and k[1] == 2 * p[1] + 1

    if not same_padding(conv.kernel_size, conv.padding):
        return None

    if "resnet" in name:
        if conv.kernel_size not in [(1, 1), (3, 3)] or conv.stride not in [
            (1, 1),
            (2, 2),
        ]:
            return None

        # width = math.gcd(conv.in_channels, conv.out_channels)
        # if width != min(conv.in_channels, conv.out_channels):
        #     return None
        if conv.groups > 1:
            return None

    elif "mobilenet_v2" in name:
        if conv.kernel_size not in [(3, 3)] or conv.stride not in [
            (1, 1),
            (2, 2),
        ]:
            return None

        # width = math.gcd(conv.in_channels, conv.out_channels)
        # if width != min(conv.in_channels, conv.out_channels):
        #     return None
        # if conv.groups > 1:
        #     return None
    else:
        raise NotImplementedError(f"{name} is not a valid model!")

    if conv.stride == (1, 1):
        return ConvPlaceholder(conv.in_channels, conv.out_channels, conv.kernel_size)
    else:
        return nn.Sequential(
            nn.AvgPool2d(*conv.stride),
            ConvPlaceholder(conv.in_channels, conv.out_channels, conv.kernel_size),
        )


def replace_conv2d_to_placeholder(module: nn.Module, model_name: str):
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            replaced_kernel = replace_conv2d_filter(child, model_name)
            if replaced_kernel is not None:
                count += 1
                setattr(module, name, replaced_kernel)
        elif len(list(child.named_children())) > 0:
            count += replace_conv2d_to_placeholder(child, model_name)
    return count


def _get_vanilla_common_model(
    name: str,
    num_classes: int,
    input_size: Tuple[int, int, int],
) -> nn.Module:
    assert hasattr(models, name), f"Could not find model {name} in torchvision"
    return getattr(models, name)(num_classes=num_classes)


class CommonModel(KASModel):
    def __init__(
        self,
        name: str,
        num_classes: int,
        input_size: Tuple[int, int, int] = (3, 224, 224),
    ) -> None:
        super().__init__()

        # Replace conv2d
        self.model = _get_vanilla_common_model(name, num_classes, input_size)
        count = replace_conv2d_to_placeholder(self.model, name)
        self.input_size = input_size
        logging.info(f"Replaced {count} Conv2D layers to Placeholder")

    def sample_input_shape(self, seq_len=None):
        return self.input_size

    def sampler_parameters(self, seq_len=None):
        return {
            "input_shape": "[N, C_in: unordered, H, H]",
            "output_shape": "[N, C_out: unordered, H, H]",
            "primary_specs": ["N: 0", "C_in: 2", "C_out: 4", "H: 0"],
            "coefficient_specs": ["k_1=3: 2", "k_2=7: 2", "s=2: 2", "g=32: 3"],
            "fixed_io_pairs": [(0, 0)],
        }

    def forward(self, x):
        return self.model(x)


def get_common_model_args(args) -> Dict:
    assert args.model.startswith("torchvision/")
    return {
        "name": args.model[len("torchvision/") :],
        "num_classes": args.num_classes,
        "input_size": args.input_size,
    }


def get_vanilla_common_model(args) -> nn.Module:
    return _get_vanilla_common_model(**get_common_model_args(args))


def get_common_model(args) -> KASModel:
    return CommonModel(**get_common_model_args(args))
