import math
import torch
import logging
from torch import nn
from typing import Dict, Tuple, Union, List

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

class AverageMeter:
    def __init__(self, support_std=False) -> None:
        self.sum: float = 0
        if support_std:
            self.sumsq: float = 0
        self.N: int = 0
        self.support_std: bool = support_std
    
    def __eq__(self, __value: 'AverageMeter') -> bool:
        assert isinstance(__value, AverageMeter)
        if not self.support_std == __value.support_std: 
            return False
        eq_flag = self.sum == __value.sum and self.N == __value.N
        if self.support_std: 
            eq_flag = eq_flag and self.sumsq == __value.sumsq
        return eq_flag
    
    def update(self, val: float) -> None:
        self.sum += val
        if self.support_std:
            self.sumsq += val * val
        self.N += 1
    
    @property
    def mean(self) -> float:
        if self.N == 0: 
            return 0
        else:
            return self.sum / self.N
    
    @property
    def std(self) -> float:
        assert self.support_std, "std is not supported"
        if self.N <= 1 or self.sumsq / self.N - self.mean * self.mean <= 0:
            return 0
        else:
            return math.sqrt(self.sumsq / self.N - self.mean * self.mean)
    
    def serialize(self) -> List:
        if self.support_std:
            return [True, [self.sum, self.sumsq, self.N]]
        else:
            return [False, [self.sum, self.N]]

    def load(self, serial: Tuple[bool, Union[List, List]]) -> None:
        std_flag, state = serial
        if self.empty():
            self.support_std = std_flag
            if std_flag:
                self.sum, self.sumsq, self.N = state
            else:
                self.sum, self.N = state
        else:
            assert self.support_std == std_flag, "g_rave inconsistency found!"
            if std_flag:
                assert [self.sum, self.sumsq, self.N] == state, "g_rave inconsistency found!"
            else:
                assert [self.sum, self.N] == state, "g_rave inconsistency found!"
    
    def empty(self) -> bool:
        if self.support_std:
            return self.sum == 0 and self.sumsq == 0 and self.N == 0
        else:
            return self.sum == 0 and self.N == 0
    
    def __repr__(self) -> str:
        if self.support_std:
            return f"AverageMeter(mean={self.mean}, std={self.std}, N={self.N})"
        else:
            return f"AverageMeter(mean={self.mean}, N={self.N})"
        

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
