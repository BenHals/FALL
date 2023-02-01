from typing import List

from river.base.typing import ClfTarget

from fall.concept_representations import ConceptRepresentation
from fall.concept_representations.rolling_stats import RollingTimeseries
from fall.utils import Observation

from .meta_feature_distributions import BaseDistribution, GaussianDistribution


class FingerprintRepresentation(ConceptRepresentation):
    """A concept representation which represents a concept
    using the error rate of a given classifier over a recent window of size w.
    With zero observations, we default to an error rate of 0.0 to represent maximum performance.
    This is a common (implied) comparison target when testing error_rate."""

    def __init__(self, window_size: int, concept_id: int, mode: str = "active", update_period: int = 1):
        super().__init__(window_size, concept_id, mode, update_period)

        self.observed_stats_y = RollingTimeseries(window_size)
        self.observed_stats_p = RollingTimeseries(window_size)
        self.observed_stats_e = RollingTimeseries(window_size)
        self.observed_stats_features: dict[str, RollingTimeseries] = {}
        self.feature_names: list[str] = []

        self.n_meta_featues: int = -1
        self.meta_feature_values: list[float] = []
        # for active we want to remember only updates over the last window
        # otherwise, we want to remember all updates.
        self.meta_feature_distributions: list[BaseDistribution] = []

        self.target_map: dict[ClfTarget, float] = {}

        self.initialized = False

    def initialize(self, observation: Observation) -> None:
        """Initialize with information specific to the observation schema.
        We do this when we first encounter an observation.
        """
        self.n_features = len(observation.x)
        for feature_name in observation.x:
            self.observed_stats_features[feature_name] = RollingTimeseries(window_size=self.window_size)
            self.feature_names.append(feature_name)

        # We have all metafeatures calculated by the RollingTimeseries, for y, p, e and each feature.
        self.n_meta_featues = len(self.observed_stats.observed_stats_y) * (3 + len(self.feature_names))
        self.meta_feature_values = [0.0] * self.n_meta_featues
        # for active we want to remember only updates over the last window
        # otherwise, we want to remember all updates.
        self.meta_feature_distributions = [
            GaussianDistribution(memory_size=1 if self.mode == "active" else -1) for i in range(self.n_meta_featues)
        ]
        self.initialized = True

    def update_supervised(self) -> None:
        while self.new_supervised:
            new_sup_ob = self.new_supervised.popleft()
            if not self.initialized:
                self.initialize(new_sup_ob)

            y = self.target_map.setdefault(new_sup_ob.y, float(len(self.target_map)))
            p = self.target_map.setdefault(new_sup_ob.predictions[self.concept_id], float(len(self.target_map)))
            new_is_correct = p == y

            self.observed_stats_y.update(y)
            self.observed_stats_p.update(p)
            self.observed_stats_e.update(float(new_is_correct))
            for f in self.feature_names:
                self.observed_stats_features[f].update(new_sup_ob.x[f])

            # We need to store the correctness from when the observation was added
            # Because if a prediction changes we may revert a different value than
            # was added, permanantly biasing the mean.
            self.supervised_window.append((new_sup_ob, new_is_correct))

        self.last_supervised_update = self.supervised_timestep

    def extract_fingerprint(self) -> list[float]:
        fingerprint: list[float] = []
        fingerprint += self.observed_stats_y.get_stats()
        fingerprint += self.observed_stats_p.get_stats()
        fingerprint += self.observed_stats_e.get_stats()
        for f in self.feature_names:
            fingerprint += self.observed_stats_features[f].get_stats()
        return fingerprint

    def integrate_fingerprint(self, fingerprint: list[float]) -> list[float]:
        for i, val in enumerate(fingerprint):
            self.meta_feature_distributions[i].learn_one(val)
            self.meta_feature_values[i] = self.meta_feature_distributions[i].mean

        return self.meta_feature_values

    def update_unsupervised(self) -> None:
        pass

    def get_values(self) -> List:
        return self.meta_feature_values
