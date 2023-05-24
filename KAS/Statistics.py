from .Bindings import StatisticsCollector


class Statistics:
    @staticmethod
    def Print():
        print(StatisticsCollector.PrintSummary())
