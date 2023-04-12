from torch import nn
from typing import List

from .KernelPack import KernelPack
from .Placeholder import Placeholder


class Modifier():
    def __init__(self) -> None:
        pass

    @staticmethod
    def FindPlaceholders(net: nn.Module) -> List[Placeholder]:
        """Find all placeholders in the network. """
        placeholders: List[Placeholder] = [
            node for node in net.modules() if isinstance(node, Placeholder)]
        return placeholders

    @staticmethod
    def KernelReplace(net: nn.Module, kernelPacks: List[KernelPack]):
        for placeholder, kernelPack in zip(__class__.FindPlaceholders(net), kernelPacks):
            placeholder.reload(kernelPack)
