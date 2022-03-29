""" Classes for representing the distribution of a meta-feature across a concept """

import abc
from enum import Enum
from math import sqrt

from river.stats import Var


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


class SingleValueDistribution(BaseDistribution):
    """A distribution containing only the most recent value."""

    def __init__(self) -> None:
        self.distribution_type = DistributionTypes.NA
        self.value: float = 0

    def learn_one(self, val: float) -> None:
        self.value = val


class GaussianDistribution(BaseDistribution):
    """A distribution containing the mean and standard deviation of values."""

    def __init__(self) -> None:
        self.distribution_type = DistributionTypes.GAUSSIAN
        self.var = Var()

    def learn_one(self, val: float) -> None:
        self.var.update(val)

    @property
    def mean(self) -> float:
        """Gaussian mean."""
        return self.var.mean.get()

    @property
    def stdev(self) -> float:
        """Gaussian standard deviation."""
        return sqrt(self.var.get())
