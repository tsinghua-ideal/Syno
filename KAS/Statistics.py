import logging

from .Bindings import StatisticsCollector


class Statistics:
    @staticmethod
    def Summary() -> str:
        return StatisticsCollector.PrintSummary()

    @staticmethod
    def PrintLog():
        logging.info(Statistics.Summary())
