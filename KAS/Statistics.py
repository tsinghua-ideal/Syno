import logging

from .Bindings import StatisticsCollector


class Statistics:
    @staticmethod
    def PrintLog():
        logging.info(StatisticsCollector.PrintSummary())
