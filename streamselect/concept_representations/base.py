""" Abstract base concept representation and comparison. """

import abc
import typing
from typing import List

from river.base import Base
from river.base.typing import ClfTarget

from .meta_feature_distributions import BaseDistribution, DistributionTypes


class ConceptRepresentation(Base, abc.ABC):
    """A base concept representation."""

    def __init__(self) -> None:
        # A concept representation represents a concept as a finite set of values, or meta-features
        self.values: List[float] = []

        # Each meta-feature has a distribution across a concept.
        self.distribution: List[BaseDistribution] = []

    @abc.abstractmethod
    def learn_one(self, x: dict, y: ClfTarget, p: typing.Union[ClfTarget, None] = None) -> None:
        """Update a concept representation with a single observation drawn from a concept,
        classified by a given classifier. Updates supervised meta-features, as in river."""

    def predict_one(self, x: dict, p: typing.Union[ClfTarget, None] = None) -> None:
        """Update a concept representation with a single observation drawn from a concept,
        classified by a given classifier. Updates unsupervised meta-features, as in river."""

    @property
    def _vector(self) -> bool:
        """Whether or not a vector concept representation is stored.
        Determines which similarity values may be used."""
        return False

    @property
    def _distribution(self) -> DistributionTypes:
        """The format in which distributional information is stored
        Determines which feature selection methods may be used."""
        return DistributionTypes(0)
