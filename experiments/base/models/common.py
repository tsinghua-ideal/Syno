import logging
from torch import nn
from torchvision import models
from typing import Optional
import math

from .model import KASModel
from .placeholder import ConvPlaceholder


def replace_conv2d_filter(conv: nn.Conv2d) -> Optional[nn.Module]:
    # TODO: maybe relax the requirements
    
    def same_padding(k, p):
        return k[0] == 2 * p[0] + 1 and k[1] == 2 * p[1] + 1
    
    if conv.kernel_size not in [(1, 1), (3, 3), (5, 5), (7, 7)] or conv.stride not in [(1, 1), (2, 2)]:
        return None
    if not same_padding(conv.kernel_size, conv.padding):
        return None
    
    # width = math.gcd(conv.in_channels, conv.out_channels)
    # if width != min(conv.in_channels, conv.out_channels):
    #     return None
    if conv.groups > 1:
        return None
    
    if conv.stride == (1, 1):
        return ConvPlaceholder(conv.in_channels, conv.out_channels, conv.kernel_size)
    else:
        return nn.Sequential(
            nn.AvgPool2d(*conv.stride), 
            ConvPlaceholder(conv.in_channels, conv.out_channels, conv.kernel_size)
        )
        


def replace_conv2d_to_placeholder(module: nn.Module):
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            replaced_kernel = replace_conv2d_filter(child)
            if replaced_kernel is not None:
                count += 1
                setattr(module, name, replaced_kernel)
        elif len(list(child.named_children())) > 0:
            count += replace_conv2d_to_placeholder(child)
    return count


class CommonModel(KASModel):
    def __init__(self, name, num_classes) -> None:
        super().__init__()
        assert hasattr(models, name), f"Could not find model {name} in torchvision"

        # Replace conv2d
        self.model = getattr(models, name)(num_classes=num_classes)
        count = replace_conv2d_to_placeholder(self.model)
        logging.info(f"Replaced {count} Conv2D layers to Placeholder")

    def sample_input_shape(self, seq_len=None):
        return (3, 224, 224)
    
    def sampler_parameters(self, seq_len=None):
         return {
            'input_shape': '[N, C_in: unordered, H, W]',
            'output_shape': '[N, C_out: unordered, H, W]',
            'primary_specs': ['N: 0', 'C_in: 2', 'C_out: 4', 'H: 0', 'W: 0'],
            'coefficient_specs': ['k_1=3: 3', 'k_2=5: 3', 's=2: 2', 'g=32: 4'],
            'fixed_io_pairs': [(0, 0)],
        }

    def forward(self, x):
        return self.model(x)


def get_common_model(args):
    assert args.model.startswith("torchvision/")
    return CommonModel(args.model[len("torchvision/"):], args.num_classes)
