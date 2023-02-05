""" Abstract base concept representation and comparison. """

import abc
from collections import deque
from typing import Deque, List, Tuple

from river.base import Base

from fall.utils import Observation

from .meta_feature_distributions import BaseDistribution, DistributionTypes
from .normalizer import MetaFeatureNormalizer


class ConceptRepresentation(Base, abc.ABC):
    """A base concept representation."""

    def __init__(
        self,
        window_size: int,
        concept_id: int,
        normalizer: MetaFeatureNormalizer,
        mode: str = "active",
        update_period: int = 1,
    ) -> None:
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

        update_on_supervised: bool
            Default: True
            Whether or not to update the concept representation based on supervised updates

        update_on_unsupervised: bool
            Default: True
            Whether or not to update the concept representation based on unsupervised updates
        """
        self.window_size: int = window_size
        self.concept_id = concept_id
        self.normalizer = normalizer
        self.mode: str = mode
        self.update_period: int = update_period
        self.update_on_supervised: bool = True
        self.update_on_unsupervised: bool = False

        self.updates_per_window = window_size // update_period

        self.supervised_window: Deque[Tuple[Observation, bool]] = deque(maxlen=self.window_size)
        self.new_supervised: Deque[Observation] = deque()
        self.unsupervised_window: Deque[Tuple[Observation, bool]] = deque(maxlen=self.window_size)
        self.new_unsupervised: Deque[Observation] = deque()
        # Assumes the first observation will be at timestep 0.0
        self.supervised_timestep = -1.0
        self.unsupervised_timestep = -1.0
        self.last_supervised_update = -1.0
        self.last_unsupervised_update = -1.0
        self.last_supervised_concept_update = -1.0
        self.last_unsupervised_concept_update = -1.0
        self.last_classifier_evolution_timestep = -1.0

        # A concept representation represents a concept as a finite set of values, or meta-features
        self.meta_feature_values: List[float] = []
        self.classifier_meta_feature_indexs: list[int] = []

        # Each meta-feature has a distribution across a concept.
        self.meta_feature_distributions: List[BaseDistribution] = []

        # We may need to initialize after seeing an observation to learn correct dimensions
        self.initialized = False

    def initialize(self, observation: Observation) -> None:
        self.initialized = True

    def learn_one(self, supervised_observation: Observation) -> None:
        """Update a concept representation with a single observation drawn from a concept,
        classified by a given classifier. Updates supervised meta-features, as in river."""
        self.new_supervised.append(supervised_observation)
        self.supervised_timestep = supervised_observation.seen_at
        self.update_supervised()

        # If required, extract a fingerprint and use it to update the concept
        if self.update_on_supervised:
            if self.last_supervised_update >= self.last_supervised_concept_update + self.update_period:
                current_fingerprint = self.extract_fingerprint()
                self.normalizer.learn_one(current_fingerprint)
                self.integrate_fingerprint(current_fingerprint)
                self.last_supervised_concept_update = self.last_supervised_update

    def predict_one(self, unsupervised_observation: Observation) -> None:
        """Update a concept representation with a single observation drawn from a concept,
        classified by a given classifier. Updates unsupervised meta-features, as in river."""
        self.new_unsupervised.append(unsupervised_observation)
        self.unsupervised_timestep = unsupervised_observation.seen_at
        self.update_unsupervised()

        # If required, extract a fingerprint and use it to update the concept
        if self.update_on_unsupervised:
            if self.last_unsupervised_update >= self.last_unsupervised_concept_update + self.update_period:
                current_fingerprint = self.extract_fingerprint()
                self.normalizer.learn_one(current_fingerprint)
                self.integrate_fingerprint(current_fingerprint)
                self.last_unsupervised_concept_update = self.last_unsupervised_update

    def overall_normalize(self, meta_features: list[float]) -> list[float]:
        """Min max normalize using the global distribution."""
        return self.normalizer.min_max_normalize(meta_features)

    def overall_standardize(self, meta_features: list[float]) -> list[float]:
        """Standardize using the global distribution."""
        return self.normalizer.standardize(meta_features)

    def local_normalize(self, meta_features: list[float]) -> list[float]:
        """Min max normalize using the local distribution."""
        transformed_meta_features = []
        for i, mf in enumerate(meta_features):
            transformed_meta_features.append(self.meta_feature_distributions[i].min_max_normalize(mf))

        return transformed_meta_features

    def local_standardize(self, meta_features: list[float]) -> list[float]:
        """Standardize using the local distribution."""
        transformed_meta_features = []
        for i, mf in enumerate(meta_features):
            transformed_meta_features.append(self.meta_feature_distributions[i].standardize(mf))

        return transformed_meta_features

    @abc.abstractmethod
    def update_supervised(self) -> None:
        """Update supervised meta-features in an online manner.
        The idea is that we can cheaply maintain a record of meta-features across the window
        in a streaming way, so that extracting a fingerprint is cheap when needed."""

    @abc.abstractmethod
    def update_unsupervised(self) -> None:
        """Update unsupervised meta-features.
        The idea is that we can cheaply maintain a record of meta-features across the window
        in a streaming way, so that extracting a fingerprint is cheap when needed."""

    @abc.abstractmethod
    def extract_fingerprint(self) -> list[float]:
        """Extracts a representation of the current observations in the window.
        Returns the representation of the current window."""

    @abc.abstractmethod
    def integrate_fingerprint(self, fingerprint: list[float]) -> list[float]:
        """Integrates a current fingerprint into the current concept representation.
        Returns the current representation of the concept after integration."""

    @abc.abstractmethod
    def get_values(self) -> List:
        """Return a single value describing each meta-feature in the representation.
        Returned as a vector, even for single meta-feature representations."""

    def get_weight_prior(self) -> List[float]:
        """Get priors for meta-feature weights."""
        weight_priors = [1.0] * len(self.meta_feature_values)

        # Reduce weight on supervised meta-features for a period after an evolution
        time_since_last_evolution = self.last_supervised_concept_update - self.last_classifier_evolution_timestep
        if time_since_last_evolution < self.window_size * 5:
            for i in self.classifier_meta_feature_indexs:
                weight_priors[i] = time_since_last_evolution / self.window_size * 5

        return weight_priors

    def handle_classifier_evolution(self) -> None:
        """Handle changes in behaviour due to a classifier evolution.
        For example, supervised meta-features may need to be reset."""
        self.last_classifier_evolution_timestep = self.supervised_timestep

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

    @property
    def counts(self) -> list[int]:
        return [d.count for d in self.meta_feature_distributions]

    @property
    def stdevs(self) -> list[float]:
        return [d.stdev for d in self.meta_feature_distributions]
