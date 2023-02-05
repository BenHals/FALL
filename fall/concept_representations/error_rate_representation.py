from typing import List

from river.stats import Mean

from fall.concept_representations import ConceptRepresentation

from .meta_feature_distributions import GaussianDistribution
from .normalizer import MetaFeatureNormalizer


class ErrorRateRepresentation(ConceptRepresentation):
    """A concept representation which represents a concept
    using the error rate of a given classifier over a recent window of size w.
    With zero observations, we default to an error rate of 0.0 to represent maximum performance.
    This is a common (implied) comparison target when testing error_rate."""

    def __init__(
        self,
        window_size: int,
        concept_id: int,
        normalizer: MetaFeatureNormalizer,
        mode: str = "active",
        update_period: int = 1,
    ):
        super().__init__(window_size, concept_id, normalizer, mode, update_period)
        self.window_error_rate = Mean()
        self.meta_feature_values = [0.0]
        self.classifier_meta_feature_indexs = [1]
        # for active we want to remember only updates over the last window
        # otherwise, we want to remember all updates.
        self.meta_feature_distributions = [GaussianDistribution(memory_size=1 if mode == "active" else -1)]

    def update_supervised(self) -> None:
        while self.new_supervised:
            new_sup_ob = self.new_supervised.popleft()
            if not self.initialized:
                self.initialize(new_sup_ob)
            new_is_correct = new_sup_ob.predictions[self.concept_id] == new_sup_ob.y
            self.window_error_rate.update(0 if new_is_correct else 1)
            if len(self.supervised_window) >= self.window_size:
                _, discard_is_correct = self.supervised_window[0]
                self.window_error_rate.revert(0 if discard_is_correct else 1)

            # We need to store the correctness from when the observation was added
            # Because if a prediction changes we may revert a different value than
            # was added, permanantly biasing the mean.
            self.supervised_window.append((new_sup_ob, new_is_correct))

        self.last_supervised_update = self.supervised_timestep

    def extract_fingerprint(self) -> list[float]:
        avg_error_rate = self.window_error_rate.get()
        return [avg_error_rate]

    def integrate_fingerprint(self, fingerprint: list[float]) -> list[float]:
        avg_error_rate = fingerprint[0]
        self.meta_feature_distributions[0].learn_one(avg_error_rate)
        self.meta_feature_values[0] = self.meta_feature_distributions[0].mean

        return self.meta_feature_values

    def update_unsupervised(self) -> None:
        pass

    def get_values(self) -> List:
        return self.meta_feature_values
