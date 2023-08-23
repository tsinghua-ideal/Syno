import logging
import torchvision
from torch import nn
from torchvision import models

from .model import KASModel
from .placeholder import ConvPlaceholder


def replace_conv2d_filter(conv: nn.Conv2d) -> bool:
    # TODO: maybe relax the requirements
    if conv.kernel_size not in [(1, 1), (3, 3), (5, 5), (7, 7)]:
        return False
    if conv.stride != (1, 1):
        return False
    if conv.kernel_size == (1, 1) and conv.padding != (0, 0):
        return False
    if conv.kernel_size == (3, 3) and conv.padding != (1, 1):
        return False
    if conv.kernel_size == (5, 5) and conv.padding != (2, 2):
        return False
    if conv.kernel_size == (7, 7) and conv.padding != (3, 3):
        return False
    return True


def replace_conv2d_to_placeholder(module: nn.Module):
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            if replace_conv2d_filter(child):
                count += 1
                setattr(module, name, 
                        ConvPlaceholder(child.in_channels, child.out_channels, child.kernel_size))
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

    @staticmethod
    def sample_input_shape():
        return (3, 32, 32)
    
    @staticmethod
    def sampler_parameters():
         return {
            'input_shape': '[N, C_in, H, W]',
            'output_shape': '[N, C_out, H, W]',
            'primary_specs': ['N: 0', 'C_in: 2', 'C_out: 2', 'H: 2', 'W: 2'],
            'coefficient_specs': ['k=3: 6', 's=2: 4'],
            'fixed_io_pairs': [(0, 0)],
        }

    def forward(self, x):
        return self.model(x)


def get_common_model(args):
    assert args.model.startswith("torchvision/")
    return CommonModel(args.model[len("torchvision/"):], args.num_classes)
