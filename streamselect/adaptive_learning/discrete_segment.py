""" A simple discrete segment adaptive learning system. """

from typing import Callable, Optional

from river.base import Classifier, DriftDetector

from streamselect.adaptive_learning import (
    BaseAdaptiveLearner,
    BaseBufferedAdaptiveLearner,
    get_increasing_buffer_scheduler,
)
from streamselect.concept_representations import ConceptRepresentation
from streamselect.repository import RepresentationComparer, ValuationPolicy
from streamselect.states import State
from streamselect.utils import Observation


class DiscreteSegmentAL(BaseAdaptiveLearner):
    """A discrete segment adaptive learning system considers each
    data stream segment with a contiguous concept to be distict from
    and previous segment. This means previous states are always irrelevant
    and adaptation can simply be constructing a new classifier.
    As we do not need to consider previous states, we can set the repository size to 1."""

    def __init__(
        self,
        classifier_constructor: Callable[[], Classifier],
        representation_constructor: Callable[[int, int, str, int], ConceptRepresentation],
        representation_comparer: RepresentationComparer,
        drift_detector_constructor: Callable[[], DriftDetector],
        representation_update_period: int = 1,
        train_representation: bool = True,
        window_size: int = 1,
        drift_detection_mode: str = "any",
    ) -> None:
        """
        Parameters
        ----------
        classifier_constructor: Callable[[], Classifier]
            A function to generate a new classifier.

        representation_constructor: Callable[[int, int, str, int], ConceptRepresentation]
            A function to generate a new concept representation taking in:
            window_size, state_id, mode and update_period.

        representation_comparer: RepresentationComparer
            An object capable of calculating similarity between two representations.

        drift_detector_constructor: Callable[[], DriftDetector]
            A function to generate an object capable of detecting drift in a univariate stream.

        representation_update_period: int
            Default: 1
            The number of timesteps between representation updates.

        train_representation: bool
            Whether or not new states train representations.
            Must be set to automatically construct states.

        window_size: int
            Default: 1
            The number of observations to construct a concept representation over.

        drift_detection_mode: str["any", "lower", "higher]
            Default: "any"
            How change is interpreted as concept drift.
            "any": Any significant change in relevance is detected as drift.
            "lower": Significant changes where the new value is lower than the mean is detected.
            "higher": Significant changes where the new value is higher than the mean is detected.
        """

        super().__init__(
            classifier_constructor=classifier_constructor,
            representation_constructor=representation_constructor,
            representation_comparer=representation_comparer,
            drift_detector_constructor=drift_detector_constructor,
            representation_update_period=representation_update_period,
            max_size=1,
            valuation_policy=ValuationPolicy.FIFO,
            train_representation=train_representation,
            window_size=window_size,
            construct_pair_representations=False,
            prediction_mode="active",
            background_state_mode=None,
            drift_detection_mode=drift_detection_mode,
        )


class BufferedDiscreteSegmentAL(BaseBufferedAdaptiveLearner):
    """A discrete segment adaptive learning system considers each
    data stream segment with a contiguous concept to be distict from
    and previous segment. This means previous states are always irrelevant
    and adaptation can simply be constructing a new classifier.
    As we do not need to consider previous states, we can set the repository size to 1.
    Uses a buffer to delay learning from new observations."""

    def __init__(
        self,
        classifier_constructor: Callable[[], Classifier],
        representation_constructor: Callable[[int, int, str, int], ConceptRepresentation],
        representation_comparer: RepresentationComparer,
        drift_detector_constructor: Callable[[], DriftDetector],
        representation_update_period: int = 1,
        train_representation: bool = True,
        window_size: int = 1,
        drift_detection_mode: str = "any",
        buffer_timeout_max: float = 0.0,
        buffer_timeout_scheduler: Callable[
            [float, State, Optional[Observation]], float
        ] = get_increasing_buffer_scheduler(),
    ) -> None:
        """
        Parameters
        ----------
        classifier_constructor: Callable[[], Classifier]
            A function to generate a new classifier.

        representation_constructor: Callable[[int, int, str, int], ConceptRepresentation]
            A function to generate a new concept representation taking in:
            window_size, state_id, mode and update_period.

        representation_comparer: RepresentationComparer
            An object capable of calculating similarity between two representations.

        drift_detector_constructor: Callable[[], DriftDetector]
            A function to generate an object capable of detecting drift in a univariate stream.

        representation_update_period: int
            Default: 1
            The number of timesteps between representation updates.

        train_representation: bool
            Whether or not new states train representations.
            Must be set to automatically construct states.

        window_size: int
            Default: 1
            The number of observations to construct a concept representation over.

        drift_detection_mode: str["any", "lower", "higher]
            Default: "any"
            How change is interpreted as concept drift.
            "any": Any significant change in relevance is detected as drift.
            "lower": Significant changes where the new value is lower than the mean is detected.
            "higher": Significant changes where the new value is higher than the mean is detected.

        buffer_timeout_max: float
            The max number of timesteps to buffer new data before training.

        buffer_timeout_scheduler: Callable[[float, State, Observation], float
            Default: constant_max_buffer_scheduler
            A function to calculate the current buffer_timeout. The function is passed buffer_timeout_max,
            the active state and (optinally) the new observation.
            The default simply returns the buffer_timeout_max. An alternative is the increasing
            scheduler, which slowly increases the buffer_timeout so that a new classifier may learn.
        """

        super().__init__(
            classifier_constructor=classifier_constructor,
            representation_constructor=representation_constructor,
            representation_comparer=representation_comparer,
            drift_detector_constructor=drift_detector_constructor,
            representation_update_period=representation_update_period,
            max_size=1,
            valuation_policy=ValuationPolicy.FIFO,
            train_representation=train_representation,
            window_size=window_size,
            construct_pair_representations=False,
            prediction_mode="active",
            background_state_mode=None,
            drift_detection_mode=drift_detection_mode,
            buffer_timeout_max=buffer_timeout_max,
            buffer_timeout_scheduler=buffer_timeout_scheduler,
        )
