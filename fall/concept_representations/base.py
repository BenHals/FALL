""" Abstract base concept representation and comparison. """

import abc
from collections import deque
from typing import Deque, List, Tuple

from river.base import Base

from fall.utils import Observation

from .meta_feature_distributions import BaseDistribution, DistributionTypes


class ConceptRepresentation(Base, abc.ABC):
    """A base concept representation."""

    def __init__(self, window_size: int, concept_id: int, mode: str = "active", update_period: int = 1) -> None:
        """
        Parameters
        ----------
        window_size: int
            The number of observations to calculate the representation over.

        concept_id: int
            The id of the associated concept. Used to extract predictions.

        mode: str ["active", "concept"]
            Default: "active"
            The mode determines how previous observations affect the representation.
            The "active" mode captures a representation of the active window, a sliding
            window of the window_size most recent observations. This mode forgets observations
            which drop out of the window.
            The "concept" mode captures a representation of a concept, i.e., represents all
            previous observations drawn from a concept in finite memory. While a window of
            window_size observations is used to calculate some statistics, i.e., standard
            deviation, experience of all observations is retained.

        update_period: int
            Default: 1
            The number of steps between updating meta_feature_values based on self.window
        """
        self.window_size: int = window_size
        self.concept_id = concept_id
        self.mode: str = mode
        self.update_period: int = update_period
        self.updates_per_window = window_size // update_period

        self.supervised_window: Deque[Tuple[Observation, bool]] = deque(maxlen=self.window_size)
        self.new_supervised: Deque[Observation] = deque()
        self.unsupervised_window: Deque[Tuple[Observation, bool]] = deque(maxlen=self.window_size)
        self.new_unsupervised: Deque[Observation] = deque()
        # Assumes the first observation will be at timestep 0.0
        self.supervised_timestep = -1.0
        self.unsupervised_timestep = -1.0
        self.last_supervised_update = -1.0
        self.last_supervised_update = -1.0

        # A concept representation represents a concept as a finite set of values, or meta-features
        self.meta_feature_values: List[float] = []

        # Each meta-feature has a distribution across a concept.
        self.meta_feature_distributions: List[BaseDistribution] = []

    def learn_one(self, supervised_observation: Observation) -> None:
        """Update a concept representation with a single observation drawn from a concept,
        classified by a given classifier. Updates supervised meta-features, as in river."""
        self.new_supervised.append(supervised_observation)
        self.supervised_timestep = supervised_observation.seen_at
        self.update_supervised()

    def predict_one(self, unsupervised_observation: Observation) -> None:
        """Update a concept representation with a single observation drawn from a concept,
        classified by a given classifier. Updates unsupervised meta-features, as in river."""
        self.new_unsupervised.append(unsupervised_observation)
        self.unsupervised_timestep = unsupervised_observation.seen_at
        self.update_unsupervised()

    @abc.abstractmethod
    def update_supervised(self) -> None:
        """Update supervised meta-features."""

    @abc.abstractmethod
    def update_unsupervised(self) -> None:
        """Update unsupervised meta-features."""

    @abc.abstractmethod
    def get_values(self) -> List:
        """Return a single value describing each meta-feature in the representation.
        Returned as a vector, even for single meta-feature representations."""

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
