from typing import List

from river.stats import Mean

from streamselect.concept_representations import ConceptRepresentation

from .meta_feature_distributions import GaussianDistribution


class ErrorRateRepresentation(ConceptRepresentation):
    """A concept representation which represents a concept
    using the error rate of a given classifier over a recent window of size w.
    With zero observations, we default to an error rate of 0.0 to represent maximum performance.
    This is a common (implied) comparison target when testing error_rate."""

    def __init__(self, window_size: int, concept_id: int, mode: str = "active", update_period: int = 1):
        super().__init__(window_size, concept_id, mode, update_period)
        self.window_error_rate = Mean()
        self.meta_feature_values = [0.0]
        # for active we want to remember only updates over the last window
        # otherwise, we want to remember all updates.
        self.meta_feature_distributions = [GaussianDistribution(memory_size=1 if mode == "active" else -1)]

    def update_supervised(self) -> None:
        if self.supervised_timestep - self.last_supervised_update < self.update_period:
            return
        while self.new_supervised:
            new_sup_ob = self.new_supervised.popleft()
            is_correct = new_sup_ob.predictions[self.concept_id] == new_sup_ob.y
            self.window_error_rate.update(0 if is_correct else 1)
            if len(self.supervised_window) >= self.window_size:
                discard = self.supervised_window[0]
                is_correct = discard.predictions[self.concept_id] == discard.y
                self.window_error_rate.revert(0 if is_correct else 1)
            self.supervised_window.append(new_sup_ob)

        avg_error_rate = self.window_error_rate.get()
        self.meta_feature_distributions[0].learn_one(avg_error_rate)
        self.meta_feature_values[0] = self.meta_feature_distributions[0].mean

        self.last_supervised_update = self.supervised_timestep

    def update_unsupervised(self) -> None:
        pass

    def get_values(self) -> List:
        return self.meta_feature_values
