""" Testing concept representations. """
import numpy as np

from fall.concept_representations.error_rate_representation import (
    ErrorRateRepresentation,
    MetaFeatureNormalizer,
)
from fall.utils import Observation


def test_error_rate_representation() -> None:
    """The error rate representation represents a classifier using the predictions of a specific
    classifier. The representations uses the average error rate over a recent window as a meta-feature.
    With a window size of 1, we are looking at just the most recent observation.

    With zero observations, we default to an error rate of 0.0 to represent maximum performance.
    This is a common (implied) comparison target when testing error_rate."""

    y_vals = [0, 0, 1, 0, 1, 1, 1, 0]
    p_vals = [1, 0, 1, 0, 1, 0, 0, 1]
    concept_id = 0
    normalizer = MetaFeatureNormalizer()
    rep_1 = ErrorRateRepresentation(1, concept_id, normalizer)
    rep_1_vals = []
    rep_5 = ErrorRateRepresentation(5, concept_id, normalizer, mode="active")
    rep_5_vals = []
    for t, (y, p) in enumerate(zip(y_vals, p_vals)):
        ob = Observation(x={}, y=y, seen_at=t, active_state_id=concept_id)
        ob.add_prediction(p, concept_id)
        rep_1_vals.append(rep_1.meta_feature_values[0])
        rep_5_vals.append(rep_5.meta_feature_values[0])
        rep_1.learn_one(ob)
        rep_5.learn_one(ob)
    rep_1_vals.append(rep_1.meta_feature_values[0])
    rep_5_vals.append(rep_5.meta_feature_values[0])

    print(rep_1_vals)
    print(rep_5_vals)
    assert rep_1_vals == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    np.testing.assert_almost_equal(rep_5_vals, [0.0, 1 / 1, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 5, 2 / 5, 3 / 5])


def test_error_rate_representation_concept() -> None:
    """Test the concept mode, which accumulates the mean error rate over the window size."""

    y_vals = [0, 0, 1, 0, 1, 1, 1, 0]
    p_vals = [1, 0, 1, 0, 1, 0, 0, 1]
    concept_id = 0
    normalizer = MetaFeatureNormalizer()
    rep_1 = ErrorRateRepresentation(1, concept_id, normalizer, mode="active")
    rep_1_vals = []
    rep_5 = ErrorRateRepresentation(5, concept_id, normalizer, mode="active")
    rep_5_vals = []
    rep_c1 = ErrorRateRepresentation(1, concept_id, normalizer, mode="concept")
    rep_c1_vals = []
    rep_c5 = ErrorRateRepresentation(5, concept_id, normalizer, mode="concept")
    rep_c5_vals = []
    for t, (y, p) in enumerate(zip(y_vals, p_vals)):
        ob = Observation(x={}, y=y, seen_at=t, active_state_id=concept_id)
        ob.add_prediction(p, concept_id)
        rep_1_vals.append(rep_1.meta_feature_values[0])
        rep_5_vals.append(rep_5.meta_feature_values[0])
        rep_1.learn_one(ob)
        rep_5.learn_one(ob)
        rep_c1_vals.append(rep_c1.meta_feature_values[0])
        rep_c5_vals.append(rep_c5.meta_feature_values[0])
        rep_c1.learn_one(ob)
        rep_c5.learn_one(ob)
    rep_1_vals.append(rep_1.meta_feature_values[0])
    rep_5_vals.append(rep_5.meta_feature_values[0])
    rep_c1_vals.append(rep_c1.meta_feature_values[0])
    rep_c5_vals.append(rep_c5.meta_feature_values[0])

    print(rep_1_vals)
    print(rep_5_vals)
    print(rep_c1_vals)
    print(rep_c5_vals)
    # The active mode representations return the error rate over the window_size
    # Note: the first value is not included, it is just the uninitialized value.
    assert rep_1_vals == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    np.testing.assert_almost_equal(rep_5_vals, [0.0, 1 / 1, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 5, 2 / 5, 3 / 5])
    # The conept mode representations accumulate the error rate over the window_size
    np.testing.assert_almost_equal(rep_c1_vals, [0.0, 1 / 1, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 2 / 6, 3 / 7, 4 / 8])
    np.testing.assert_almost_equal(
        rep_c5_vals,
        [
            0.0,
            1 / 1,
            float(np.mean(rep_5_vals[1:3])),
            float(np.mean(rep_5_vals[1:4])),
            float(np.mean(rep_5_vals[1:5])),
            float(np.mean(rep_5_vals[1:6])),
            float(np.mean(rep_5_vals[1:7])),
            float(np.mean(rep_5_vals[1:8])),
            float(np.mean(rep_5_vals[1:9])),
        ],
    )


def test_error_rate_representation_update_period() -> None:
    """Test the concept mode, which accumulates the mean error rate over the window size."""

    y_vals = [0, 0, 1, 0, 1, 1, 1, 0]
    p_vals = [1, 0, 1, 0, 1, 0, 0, 1]
    concept_id = 0
    update_period = 3
    normalizer = MetaFeatureNormalizer()
    rep_1 = ErrorRateRepresentation(1, concept_id, normalizer, mode="active", update_period=update_period)
    rep_1_vals = []
    rep_5 = ErrorRateRepresentation(5, concept_id, normalizer, mode="active", update_period=update_period)
    rep_5_vals = []
    rep_c1 = ErrorRateRepresentation(1, concept_id, normalizer, mode="concept", update_period=update_period)
    rep_c1_vals = []
    rep_c5 = ErrorRateRepresentation(5, concept_id, normalizer, mode="concept", update_period=update_period)
    rep_c5_vals = []
    for t, (y, p) in enumerate(zip(y_vals, p_vals)):
        ob = Observation(x={}, y=y, seen_at=t, active_state_id=concept_id)
        ob.add_prediction(p, concept_id)
        rep_1_vals.append(rep_1.meta_feature_values[0])
        rep_5_vals.append(rep_5.meta_feature_values[0])
        rep_1.learn_one(ob)
        rep_5.learn_one(ob)
        rep_c1_vals.append(rep_c1.meta_feature_values[0])
        rep_c5_vals.append(rep_c5.meta_feature_values[0])
        rep_c1.learn_one(ob)
        rep_c5.learn_one(ob)
    rep_1_vals.append(rep_1.meta_feature_values[0])
    rep_5_vals.append(rep_5.meta_feature_values[0])
    rep_c1_vals.append(rep_c1.meta_feature_values[0])
    rep_c5_vals.append(rep_c5.meta_feature_values[0])

    print(rep_1_vals)
    print(rep_5_vals)
    print(rep_c1_vals)
    print(rep_c5_vals)
    # The update period should only update meta_feature_values every update_period steps, otherwise they should
    # be the same as the last.
    # for update_period = 3
    # [update, skip, skip, update, skip, skip, update, skip, skip,]
    assert rep_1_vals == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    np.testing.assert_almost_equal(rep_5_vals, [0.0, 0.0, 0.0, 1 / 3, 1 / 3, 1 / 3, 1 / 5, 1 / 5, 1 / 5])
    np.testing.assert_almost_equal(rep_c1_vals, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1 / 2, 1 / 2, 1 / 2])
    np.testing.assert_almost_equal(
        rep_c5_vals,
        [
            0.0,
            0.0,
            0.0,
            float(np.mean([rep_5_vals[3]])),
            float(np.mean([rep_5_vals[3]])),
            float(np.mean([rep_5_vals[3]])),
            float(np.mean([rep_5_vals[3], rep_5_vals[6]])),
            float(np.mean([rep_5_vals[3], rep_5_vals[6]])),
            float(np.mean([rep_5_vals[3], rep_5_vals[6]])),
        ],
    )
