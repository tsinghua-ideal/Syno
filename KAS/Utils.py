from .Bindings import Next
from typing import Dict, Tuple

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
    def __init__(self) -> None:
        self.sum: float = 0
        self.N: int = 0
    
    def __eq__(self, __value: 'AverageMeter') -> bool:
        assert isinstance(__value, AverageMeter)
        return self.sum == __value.sum and self.N == __value.N
    
    def update(self, val: float) -> None:
        self.sum += val
        self.N += 1
    
    @property
    def avg(self) -> float:
        if self.N == 0: return 0
        return self.sum / self.N
    
    def serialize(self) -> Dict:
        return (self.sum, self.N)

    @staticmethod
    def deserialize(serial: Tuple[float, int]) -> 'AverageMeter':
        am = AverageMeter()
        am.sum, am.N = serial
        return am
    
    def empty(self) -> bool:
        return self.sum == 0 and self.N == 0
    
    def __repr__(self) -> str:
        return f"AverageMeter(sum={self.sum}, N={self.N})"