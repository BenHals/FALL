""" A simple discrete segment adaptive learning system. """

from streamselect.adaptive_learning import BaseAdaptiveLearner


class DiscreteSegmentAL(BaseAdaptiveLearner):
    """A discrete segment adaptive learning system considers each
    data stream segment with a contiguous concept to be distict from
    and previous segment. This means previous states are always irrelevant
    and adaptation can simply be constructing a new classifier."""

    def __init__(self) -> None:
        pass
