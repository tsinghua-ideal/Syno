from typing import List, Optional
import math


class SimulationCounter:
    def __init__(
        self,
        N: int = 100,
        leaf_num: int = 1,
        increase_ratio: float = 1.5,
        retry_times: int = 5,
        upper_bound: int = 30000,
    ) -> None:
        self.N: int = N
        self.success_q: List[int] = []
        self.total_q: List[int] = []

        self.leaf_num = leaf_num
        self.ratio = increase_ratio
        self.retry = retry_times
        self.upper_bound = upper_bound

        assert self.ratio > 1

    def empty(self) -> bool:
        return len(self.success_q) == 0

    def update(self, success: int, total: int) -> None:
        assert len(self.success_q) == len(self.total_q)
        if len(self.success_q) >= self.N:
            self.success_q = self.success_q[1:]
            self.total_q = self.total_q[1:]
        self.success_q.append(success)
        self.total_q.append(total)

    def list_update(self, success: List[int], total: List[int]) -> None:
        assert len(self.success_q) == len(self.total_q)
        assert len(success) == len(total)
        if len(success) > self.N:
            success = success[-self.N :]
            total = total[-self.N :]
        if len(self.success_q) > self.N - len(success):
            self.success_q = self.success_q[len(success) :]
            self.total_q = self.total_q[len(total) :]
        self.success_q += success
        self.total_q += total

    def estimated_probability(self) -> Optional[float]:
        if self.empty():
            return 0
        else:
            return sum(self.success_q) / sum(self.total_q)

    def get_trials(self):
        if self.estimated_probability() == 0:

            def trial_gen():
                trials = self.leaf_num
                while trials <= self.upper_bound:
                    yield trials
                    trials = math.ceil(trials * self.ratio)

            return trial_gen()
        else:
            assert self.estimated_probability() > 0

            def trial_gen():
                trials = math.ceil(self.leaf_num / self.estimated_probability())
                for _ in range(self.retry):
                    yield trials
                    trials = math.ceil(trials * self.ratio)

            return trial_gen()
