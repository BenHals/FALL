""" Tests for meta-feature distributions. """
import numpy as np

from streamselect.concept_representations.meta_feature_distributions import (
    GaussianDistribution,
    SingleValueDistribution,
)


def test_single_value_distribution() -> None:
    """The single value distribution should always return only the most recently assigned value."""
    test_meta_feature_vals = [0, 1, 2, 3, 2, 5, 0]
    mf_distribution = SingleValueDistribution()
    output = []
    for val in test_meta_feature_vals:
        output.append(mf_distribution.value)
        mf_distribution.learn_one(val)
    output.append(mf_distribution.value)

    print(output)
    assert output == [0, *[0, 1, 2, 3, 2, 5, 0]]


def test_gaussian_distribution() -> None:
    """The gaussian distribution should maintain a mean and standard deviation across values."""
    test_meta_feature_vals = [0, 1, 2, 3, 2, 5, 0]
    mf_distribution = GaussianDistribution()
    output = []
    for val in test_meta_feature_vals:
        output.append((mf_distribution.mean, mf_distribution.stdev))
        mf_distribution.learn_one(val)
    output.append((mf_distribution.mean, mf_distribution.stdev))

    test_against = np.array(
        [
            (0.0, 0.0),
            *[
                (
                    np.mean(test_meta_feature_vals[:i]),
                    np.std(test_meta_feature_vals[:i], ddof=1),
                )
                for i in range(1, len(test_meta_feature_vals) + 1)
            ],
        ]
    )
    # Note: we have desired behaviour that the stdev of distributions with 1 value is 0.
    test_against = np.nan_to_num(test_against)
    np.testing.assert_array_almost_equal(output, test_against, decimal=2)
