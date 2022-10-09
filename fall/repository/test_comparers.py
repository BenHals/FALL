""" Testing concept representations comparison. """
from numpy.testing import assert_array_almost_equal

from fall.concept_representations.error_rate_representation import (
    ErrorRateRepresentation,
)
from fall.repository.comparers import AbsoluteValueComparer
from fall.utils import Observation


def test_absolute_value_comparer() -> None:
    """Test comparing absolute values."""

    # It is common to assume a baseline error rate of 1.0
    # This is the default.
    concept_id = 0
    baseline = ErrorRateRepresentation(1, concept_id)
    y_vals = [0, 0, 1, 0, 1, 1, 1, 0]
    p_vals = [1, 0, 1, 0, 1, 0, 0, 1]
    rep_1 = ErrorRateRepresentation(1, concept_id)
    rep_5 = ErrorRateRepresentation(5, concept_id)
    sim = AbsoluteValueComparer()
    rep_1_similarity = []
    rep_5_similarity = []
    for t, (y, p) in enumerate(zip(y_vals, p_vals)):
        ob = Observation({}, y, t, concept_id)
        ob.add_prediction(p, concept_id)
        rep_1_similarity.append(sim.get_similarity(rep_1, baseline))
        rep_5_similarity.append(sim.get_similarity(rep_5, baseline))
        rep_1.learn_one(ob)
        rep_5.learn_one(ob)
    rep_1_similarity.append(sim.get_similarity(rep_1, baseline))
    rep_5_similarity.append(sim.get_similarity(rep_5, baseline))

    print(rep_1_similarity)
    print(rep_5_similarity)

    # The first value should always be 0, because we are comparing two baseline
    # representations, i.e., seen no values.
    # For error rate, the similarity to the baseline is the inverse error rate,
    # i.e., accuracy.
    # This is because distance to error rate 0.0 gives error rate, and 1-error rate
    # gives accuracy.
    # This means absolute similarity between an error rate classifier and the perfect
    # baseline gives accuracy, i.e., maximizing similarity means selecting
    # the most accurate classifier.
    assert rep_1_similarity == [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert_array_almost_equal(rep_5_similarity, [1.0, 0 / 1, 1 / 2, 2 / 3, 3 / 4, 4 / 5, 4 / 5, 3 / 5, 2 / 5])
