""" Classes for representing the distribution of a meta-feature across a concept """

import abc
from enum import Enum
from math import sqrt

from river import utils
from river.stats import Mean, Var


class DistributionTypes(Enum):
    """Enum describing the type of distribution represented."""

    NA = 0
    GAUSSIAN = 1
    HISTOGRAM = 2
    MATRIXSKETCH = 3


class BaseDistribution(abc.ABC):
    """Represents the distribution of a meta-feature across a concept."""

    @abc.abstractmethod
    def learn_one(self, val: float) -> None:
        """Add a new value to the distribution."""

    def reset_distribution(self) -> None:
        """Reset the variance associated with the meta-feature."""

    @property
    def mean(self) -> float:
        """Returns a value describing the distribution."""
        return 0.0

    @property
    def stdev(self) -> float:
        """Returns a value describing the variance."""
        return 0.0

    def __str__(self) -> str:
        return f"|{self.mean}-{self.stdev}|"

    def __repr__(self) -> str:
        return str(self)


class SingleValueDistribution(BaseDistribution):
    """A distribution containing only the most recent value."""

    def __init__(self, memory_size: int = -1) -> None:
        self.distribution_type = DistributionTypes.NA
        self.value: float = 0

    def learn_one(self, val: float) -> None:
        self.value = val

    @property
    def mean(self) -> float:
        """Returns a value describing the distribution."""
        return self.value


class GaussianDistribution(BaseDistribution):
    """A distribution containing the mean and standard deviation of values."""

    def __init__(self, memory_size: int = -1) -> None:
        self.memory_size = memory_size
        self.distribution_type = DistributionTypes.GAUSSIAN
        self.is_rolling = self.memory_size > 0
        if self.is_rolling:
            self.var = utils.Rolling(Var(), window_size=memory_size)
            self.mean_stat = utils.Rolling(Mean(), window_size=memory_size)
        else:
            self.var = Var()
            self.mean_stat = Mean()

    def learn_one(self, val: float) -> None:
        self.var.update(val)
        self.mean_stat.update(val)

    @property
    def mean(self) -> float:
        """Gaussian mean."""
        return self.mean_stat.get()  # type: ignore

    @property
    def stdev(self) -> float:
        """Gaussian standard deviation."""
        return sqrt(self.var.get())
