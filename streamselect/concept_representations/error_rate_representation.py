from typing import Union

from river.base.typing import ClfTarget
from river.stats import RollingMean

from streamselect.concept_representations import ConceptRepresentation

from .meta_feature_distributions import DistributionTypes, SingleValueDistribution


class ErrorRateRepresentation(ConceptRepresentation):
    """A concept representation which represents a concept
    using the error rate of a given classifier over a recent window of size w."""

    def __init__(self, window_size: int):
        super().__init__()
        self.recent_error_rate = RollingMean(window_size)
        self.values = [0.0]
        self.distribution = [SingleValueDistribution()]

    def learn_one(self, x: dict, y: ClfTarget, p: Union[ClfTarget, None] = None) -> None:
        self.recent_error_rate.update(1 if p != y else 0)
        avg_error_rate = self.recent_error_rate.get()
        self.values[0] = avg_error_rate
        self.distribution[0].learn_one(avg_error_rate)

    @property
    def _vector(self) -> bool:
        return False

    @property
    def _distribution(self) -> DistributionTypes:
        return DistributionTypes.NA
