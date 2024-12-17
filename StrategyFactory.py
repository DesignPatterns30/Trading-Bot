from Strategies import (
    MomentumStrategy,
    MovingAveragesStrategy,
    AnalyzePatternStrategy,
    RSIStrategy,
    XGBStrategy,
)
from abc import ABC, abstractmethod


class StrategyCreator(ABC):  # Our abstract factory class
    @abstractmethod
    def factory_method(self):
        pass

    def create_strategy(self, metric):
        strategy = self.factory_method(metric)
        return strategy


class StrategyFactory:
    @staticmethod
    def create_strategy(metric):
        low_threshold = 150
        mid_threshold = 200
        high_threshold = 300

        if metric is None:  # We use our metric to evaluate which class to use
            creator = MovingAveragesStrategyCreator()
        elif metric > high_threshold or metric < low_threshold:
            creator = XGBStrategyCreator()
        elif mid_threshold < metric <= high_threshold:
            creator = MomentumStrategyCreator()
        elif low_threshold < metric <= mid_threshold:
            creator = RSIStrategyCreator()
        else:
            creator = AnalyzePatternStrategyCreator()

        return creator.create_strategy(metric)


class MomentumStrategyCreator(
    StrategyCreator
):  # Our concrete creators that extend the abstract StrategyCreator class
    def factory_method(self, metric):
        return MomentumStrategy()  # We return the instance of the strategy
        # Instantiation is deferred to this child class


class MovingAveragesStrategyCreator(StrategyCreator):
    def factory_method(self, metric):
        return MovingAveragesStrategy(window_size=5)


class AnalyzePatternStrategyCreator(StrategyCreator):
    def factory_method(self, metric):
        return AnalyzePatternStrategy()


class RSIStrategyCreator(StrategyCreator):
    def factory_method(self, metric):
        return RSIStrategy()


class XGBStrategyCreator(StrategyCreator):
    def factory_method(self, metric):
        return XGBStrategy()
