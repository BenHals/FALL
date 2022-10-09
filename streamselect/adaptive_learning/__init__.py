""" Adaptive learning systems which perform concept drift detection and concept
re-identification to adapt to concept drift in a streaming setting.

An adaptive learning system has three main tasks, with an overall goal of
maintaining an active state which is relevant to current streaming conditions.

1) Monitor the relevance of the current active state to recent data. (Concept drift detection.)
2) Calculate the relevance of inactive states from previous concepts to recent data. (Concept re-identification.)
3) Construct the optimal active state to handle new observations (Concept adaptation.)
"""

from .base import (
    BaseAdaptiveLearner,
    BaseBufferedAdaptiveLearner,
    get_constant_max_buffer_scheduler,
    get_increasing_buffer_scheduler,
)
from .classifier_adaptation import (
    max_acc_sig_relevance_adaptation,
    maximum_relevance_adaptation,
)
from .discrete_segment import BufferedDiscreteSegmentAL, DiscreteSegmentAL

__all__ = [
    "BaseAdaptiveLearner",
    "BaseBufferedAdaptiveLearner",
    "DiscreteSegmentAL",
    "BufferedDiscreteSegmentAL",
    "get_constant_max_buffer_scheduler",
    "get_increasing_buffer_scheduler",
    "maximum_relevance_adaptation",
    "max_acc_sig_relevance_adaptation",
]
