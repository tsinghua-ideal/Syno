import math
from typing import List, Tuple, Union


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

    def refresh(self, serial: Tuple[bool, List]) -> None:
        """
        Load or Update 
        """
        std_flag, state = serial
        if self.empty() or self.N < state[-1]:
            self.support_std = std_flag
            if std_flag:
                self.sum, self.sumsq, self.N = state
            else:
                self.sum, self.N = state
        # else:
        #     assert self.support_std == std_flag, f"g_rave inconsistency found! {std_flag} v.s. {self.support_std}"
        #     if std_flag:
        #         assert [self.sum, self.sumsq, self.N] == state, f"g_rave inconsistency found! {state} v.s. {[self.sum, self.sumsq, self.N]}"
        #     else:
        #         assert [self.sum, self.N] == state, f"g_rave inconsistency found! {state} v.s. {[self.sum, self.N]}"
    
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
