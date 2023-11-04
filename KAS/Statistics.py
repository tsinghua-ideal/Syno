import logging
from typing import Optional

from .Bindings import StatisticsCollector
from .Sampler import Sampler


class Statistics:
    @staticmethod
    def Summary(sampler: Optional[Sampler] = None) -> str:
        return StatisticsCollector.PrintSummary() + (f"\n{sampler.get_all_stats()}" if sampler is not None else "")

    @staticmethod
    def PrintLog(sampler: Optional[Sampler] = None):
        logging.info(Statistics.Summary(sampler))
