from torch import nn
from typing import Dict, List

from .Bindings import Next
from .KernelPack import KernelPack


class NextSerializer:
    def __init__(self):
        next_type_cnt = Next.NumTypes
        self._next_type_to_str = {}
        self._str_to_next_type = {}
        for i in range(next_type_cnt):
            next_type = Next.Type(i)
            name = str(next_type).split('.')[-1]
            self._next_type_to_str[next_type] = name
            self._str_to_next_type[name] = next_type

    def serialize_type(self, next_type: Next.Type) -> str:
        return self._next_type_to_str[next_type]

    def deserialize_type(self, next_type: str) -> Next.Type:
        return self._str_to_next_type[next_type]

    def serialize_next(self, next: Next) -> str:
        return f"{next}"

    def deserialize_next(self, next: str) -> Next:
        next_type, key = next.split('(')
        next_type = self.deserialize_type(next_type)
        key = int(key[:-1])
        return Next(next_type, key)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.)
        nn.init.constant_(m.bias, 0.)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight.data, std=.1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, KernelPack):
        for w in m.weights:
            nn.init.trunc_normal_(w, std=.1)
