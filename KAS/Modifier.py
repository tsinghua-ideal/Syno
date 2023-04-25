from torch import nn
from typing import List

from .KernelPack import KernelPack
from .Placeholder import Placeholder


class Modifier:
    
    @staticmethod
    def FindPlaceholders(net: nn.Module) -> List[Placeholder]:
        """Find all placeholders in the network. """
        placeholders: List[Placeholder] = [
            node for node in net.modules() if isinstance(node, Placeholder)]
        return placeholders

    @staticmethod
    def KernelReplace(placeholders: List[Placeholder], kernelPacks: List[KernelPack]):
        for placeholder, kernelPack in zip(placeholders, kernelPacks):
            placeholder.reload(kernelPack)
