""" Testing concept representations. """

from streamselect.concept_representations.error_rate_representation import (
    ErrorRateRepresentation,
)


def test_error_rate_representation() -> None:
    """The error rate representation represents a classifier using the predictions of a specific
    classifier. The representations uses the average error rate over a recent window as a meta-feature.
    With a window size of 1, we are looking at just the most recent observation.

    With zero observations, we default to an error rate of 0.0 to represent maximum performance.
    This is a common (implied) comparison target when testing error_rate."""

    y_vals = [0, 0, 1, 0, 1, 1, 1, 0]
    p_vals = [1, 0, 1, 0, 1, 0, 0, 1]
    rep_1 = ErrorRateRepresentation(1)
    rep_1_vals = []
    rep_5 = ErrorRateRepresentation(5)
    rep_5_vals = []
    for y, p in zip(y_vals, p_vals):
        rep_1_vals.append(rep_1.values[0])
        rep_5_vals.append(rep_5.values[0])
        rep_1.learn_one({}, y, p)
        rep_5.learn_one({}, y, p)
    rep_1_vals.append(rep_1.values[0])
    rep_5_vals.append(rep_5.values[0])

    print(rep_1_vals)
    print(rep_5_vals)
    assert rep_1_vals == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert rep_5_vals == [0.0, 1 / 1, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 5, 2 / 5, 3 / 5]
