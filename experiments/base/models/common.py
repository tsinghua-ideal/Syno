import logging
from torch import nn
from torchvision import models, ops
from typing import Dict, Optional, Tuple
import math

from .model import KASModel
from .placeholder import (
    ConvPlaceholder,
    ViTLinearPlaceholder,
    ConvNeXtLinearPlaceholder,
    Conv3dPlaceholder
)


def replace_layer_filter(
    layer: nn.Conv2d | ops.misc.MLP | nn.Conv3d, name: str
) -> Optional[nn.Module]:
    # TODO: maybe relax the requirements

    if isinstance(layer, ops.misc.MLP) and "vit" in name:
        for name, child in layer.named_children():
            if isinstance(child, nn.Linear):
                setattr(
                    layer,
                    name,
                    ViTLinearPlaceholder(child.in_features, child.out_features),
                )
        return layer

    elif isinstance(layer, models.convnext.CNBlock) and "convnext_tiny" in name:
        for name, child in layer.block.named_children():
            if isinstance(child, nn.Linear):
                setattr(
                    layer.block,
                    name,
                    ConvNeXtLinearPlaceholder(child.in_features, child.out_features),
                )
        return layer

    elif isinstance(layer, nn.Conv2d) and ("video" not in name):

        def same_padding(k, p):
            return k[0] == 2 * p[0] + 1 and k[1] == 2 * p[1] + 1

        if not same_padding(layer.kernel_size, layer.padding):
            return None

        if "resnet" in name:
            if layer.kernel_size not in [(1, 1), (3, 3)] or layer.stride not in [
                (1, 1),
                (2, 2),
            ]:
                return None
            if layer.groups > 1:
                return None

        elif "mobilenet_v2" in name:
            if layer.kernel_size not in [(3, 3)] or layer.stride not in [
                (1, 1),
                (2, 2),
            ]:
                return None

        elif "densenet" in name:
            if layer.kernel_size not in [(3, 3)] or layer.stride not in [
                (1, 1),
                (2, 2),
            ]:
                return None

        elif "resnext29_2x64d" in name:
            if layer.kernel_size not in [(3, 3)] or layer.stride not in [
                (1, 1),
                (2, 2),
            ]:
                return None

        elif "efficientnet" in name:
            if layer.kernel_size not in [(3, 3)] or layer.stride not in [
                (1, 1),
                (2, 2),
            ]:
                return None
            if layer.groups > 1:
                return None

        elif "convnext_tiny" in name:
            # if layer.kernel_size not in [(3, 3)] or layer.stride not in [
            #     (1, 1),
            #     (2, 2),
            # ]:
            #     return None
            # if layer.groups > 1:
            #     return None
            return None
        else:
            raise NotImplementedError(f"{name} is not a valid model!")

        if layer.stride == (1, 1):
            return ConvPlaceholder(
                layer.in_channels, layer.out_channels, layer.kernel_size, layer.groups
            )
        else:
            return nn.Sequential(
                nn.AvgPool2d(*layer.stride),
                ConvPlaceholder(
                    layer.in_channels,
                    layer.out_channels,
                    layer.kernel_size,
                    layer.groups,
                ),
            )
        
    elif isinstance(layer, nn.Conv3d) and ("video" in name):
        
        if "r3d" in name:
            if layer.kernel_size not in [(1, 1, 1), (3, 3, 3)] or layer.stride not in [
                (1, 1, 1),
                (2, 2, 2),
            ]:
                return None
            if layer.groups > 1:
                return None
        else:
            raise NotImplementedError(f"{name} is not a valid model!")

        if layer.stride == (1, 1, 1):
            return Conv3dPlaceholder(
                layer.in_channels, layer.out_channels, layer.kernel_size, layer.groups
            )
        else:
            return nn.Sequential(
                nn.AvgPool3d(kernel_size=layer.stride, stride=layer.stride),
                Conv3dPlaceholder(
                    layer.in_channels,
                    layer.out_channels,
                    layer.kernel_size,
                    layer.groups,
                ),
            )
    else:
        return None


def replace_layers_to_placeholder(module: nn.Module, model_name: str):
    count = 0
    for name, child in module.named_children():
        replaced_kernel = replace_layer_filter(child, model_name)
        if replaced_kernel is not None:
            count += 1
            setattr(module, name, replaced_kernel)
        if len(list(child.named_children())) > 0:
            count += replace_layers_to_placeholder(child, model_name)
    return count


def _get_vanilla_common_model(
    name: str,
    num_classes: int,
    input_size: Tuple[int, int, int],
    temporal_size: int
) -> nn.Module:
    if name == "resnext29_2x64d":
        return models.resnet.ResNet(
            models.resnet.Bottleneck, [3, 3, 3, 0], groups=2, width_per_group=64
        )
    if name.startswith("video/"):
        name = name[len("video/"):]
        assert hasattr(models.video, name), f"Could not find model {name} in torchvision.video"
        assert temporal_size, "Temporal size is not assigned. "
        return getattr(models.video, name)(num_classes=num_classes)
    assert hasattr(models, name), f"Could not find model {name} in torchvision"
    return getattr(models, name)(num_classes=num_classes)


class CommonModel(KASModel):
    def __init__(
        self,
        name: str,
        num_classes: int,
        input_size: Tuple[int, int, int] = (3, 224, 224),
        temporal_size: int = 16
    ) -> None:
        super().__init__()

        # Replace conv2d
        self.model = _get_vanilla_common_model(name, num_classes, input_size, temporal_size)
        C, H, W = input_size
        self.input_size = (C, temporal_size, H, W) if "video" in name else input_size
        self.name = name
        count = replace_layers_to_placeholder(self.model, name)
        logging.info(f"Replaced {count} Conv layers to Placeholder")

    def sample_input_shape(self, seq_len=None):
        return self.input_size

    def sampler_parameters(self, seq_len=None):
        if "vit" in self.name:
            return {
                "input_shape": "[N, seq_len, H_in: unordered]",
                "output_shape": "[N, seq_len, H_out: unordered]",
                "primary_specs": ["N: 0", "seq_len: 0", "H_in: 2", "H_out: 3"],
                "coefficient_specs": ["k_1=3: 3", "s=2: 3", "g=384: 3"],
                "fixed_io_pairs": [(0, 0)],
            }
        elif "video" in self.name:
            return {
                "input_shape": "[N, C_in: unordered, T, H, H]",
                "output_shape": "[N, C_out: unordered, T, H, H]",
                "primary_specs": ["N: 0", "C_in: 3", "C_out: 4", "T: 0", "H: 0"],
                "coefficient_specs": ["k_1=3: 2", "k_2=7: 2", "s=2: 2", "g=32: 3"],
                "fixed_io_pairs": [(0, 0)],
            }
        else:
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
        "temporal_size": args.temporal_size, 
    }


def get_vanilla_common_model(args) -> nn.Module:
    return _get_vanilla_common_model(**get_common_model_args(args))


def get_common_model(args) -> KASModel:
    return CommonModel(**get_common_model_args(args))
