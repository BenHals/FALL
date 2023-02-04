from functools import partial
from typing import Callable, Optional

import numpy as np

from fall.concept_representations import MetaFeatureNormalizer  # ConceptRepresentation
from fall.feature_selection.fisher_score import fisher_score
from fall.repository import Repository

# from fall.states import State


def get_n_metafeatures(repository: Repository) -> int:
    n_metafeatures = 1
    for _, state in repository.states.items():
        active_representation = state.get_self_representation()
        if active_representation is None or not active_representation.initialized:
            continue
        n_metafeatures = len(active_representation.meta_feature_values)
    return n_metafeatures


def uniform_weighting(repository: Repository, normalizer: MetaFeatureNormalizer) -> list[float]:
    n_metafeatures = get_n_metafeatures(repository)
    if n_metafeatures == 0:
        print("Warning: Metafeature dimensions cannot be determined")

    return [1.0] * n_metafeatures


def random_weighting(repository: Repository, normalizer: MetaFeatureNormalizer) -> list[float]:
    n_metafeatures = get_n_metafeatures(repository)

    if n_metafeatures == 0:
        print("Warning: Metafeature dimensions cannot be determined")

    return np.random.rand(n_metafeatures).tolist()


def fisher_overall_weighting(repository: Repository, normalizer: MetaFeatureNormalizer) -> list[float]:
    n_metafeatures = get_n_metafeatures(repository)

    if n_metafeatures == 0:
        print("Warning: Metafeature dimensions cannot be determined")

    return map_weights(repository, normalizer, partial(calc_weight, fisher_overall_weight))


def map_weights(
    repository: Repository,
    normalizer: MetaFeatureNormalizer,
    weight_calc: Callable,
    mi_formula: Optional[Callable] = None,
    weighted: bool = False,
) -> list[float]:
    """Parses a repository into numpy matricies for feature selection weighting."""
    n_metafeatures = get_n_metafeatures(repository)
    state_ids = list(repository.states.keys())
    metafeatures_by_active_states: list[np.ndarray] = []
    metafeature_stdevs_by_active_states: list[np.ndarray] = []
    metafeature_counts_by_active_states: list[np.ndarray] = []
    metafeature_initialized_by_active_states: list[np.ndarray] = []
    for i in state_ids:
        metafeatures_for_active_state_i: list[np.ndarray] = []
        metafeature_stdevs_for_active_state_i: list[np.ndarray] = []
        metafeature_counts_for_active_state_i: list[np.ndarray] = []
        metafeature_initialized_for_active_state_i: list[bool] = []
        for j in state_ids:
            # These metafeatures represent the distribution generated from state j,
            # processed using state i as the active state.
            joint_metafeatures = repository.states[i].concept_representation.get(j, None)
            metafeature_initialized_for_state_j_distribution = (
                joint_metafeatures.initialized if joint_metafeatures else False
            )
            if metafeature_initialized_for_state_j_distribution and joint_metafeatures is not None:
                metafeatures_for_state_j_distribution = np.array(joint_metafeatures.meta_feature_values)
                metafeature_stdevs_for_state_j_distribution = np.array(joint_metafeatures.stdevs)
                metafeature_counts_for_state_j_distribution = np.array(joint_metafeatures.counts)
            else:
                metafeatures_for_state_j_distribution = np.array(np.zeros(n_metafeatures))
                metafeature_stdevs_for_state_j_distribution = np.array(np.zeros(n_metafeatures))
                metafeature_counts_for_state_j_distribution = np.array(np.zeros(n_metafeatures))

            metafeatures_for_active_state_i.append(metafeatures_for_state_j_distribution)
            metafeature_stdevs_for_active_state_i.append(metafeature_stdevs_for_state_j_distribution)
            metafeature_counts_for_active_state_i.append(metafeature_counts_for_state_j_distribution)
            metafeature_initialized_for_active_state_i.append(metafeature_initialized_for_state_j_distribution)
        metafeatures_by_active_states.append(np.stack(metafeatures_for_active_state_i, axis=0))
        metafeature_stdevs_by_active_states.append(np.stack(metafeature_stdevs_for_active_state_i, axis=0))
        metafeature_counts_by_active_states.append(np.stack(metafeature_counts_for_active_state_i, axis=0))
        metafeature_initialized_by_active_states.append(np.stack(metafeature_initialized_for_active_state_i, axis=0))

    # 3D matrix (i - index of the active state, j - index of the generating distribution, k - index of the metafeature)
    metafeature_matrix = np.stack(metafeatures_by_active_states, axis=0)
    metafeature_stdevs_matrix = np.stack(metafeature_stdevs_by_active_states, axis=0)
    metafeature_counts_matrix = np.stack(metafeature_counts_by_active_states, axis=0)
    # 2D matrix (i - index of the active state, j - index of the generating distribution)
    # Each element is an int in {0, 1} determining if the corresponding representation has been initialized.
    # If 0, then we should ignore the metafeature values.
    metafeature_initialized_matrix = np.stack(metafeature_initialized_by_active_states, axis=0)

    final_weights = np.zeros(n_metafeatures)
    for k in range(n_metafeatures):

        final_weights[k] = weight_calc(
            k,
            repository,
            normalizer,
            mi_formula,
            weighted,
            metafeature_matrix,
            metafeature_stdevs_matrix,
            metafeature_counts_matrix,
            metafeature_initialized_matrix,
        )

    return process_weights(final_weights)


def process_weights(weights: np.ndarray, nan_fill_val: float = 0.01) -> list[float]:
    min_val = np.nanmin(weights)
    fill_val = min_val if not np.isnan(min_val) else nan_fill_val
    return np.nan_to_num(weights, nan=fill_val).tolist()


def calc_weight(
    weight_func: Callable,
    k: int,
    repository: Repository,
    normalizer: MetaFeatureNormalizer,
    mi_formula: Optional[Callable],
    weighted: bool,
    metafeature_matrix: np.ndarray,
    metafeature_stdevs_matrix: np.ndarray,
    metafeature_counts_matrix: np.ndarray,
    metafeature_initialized_matrix: np.ndarray,
) -> float:
    """General process for calculating a feature weight.
    Features are weighed independently.
    This does not capture covariance, so
    cannot find 'redundant' features!
    Features with high correlation should
    not be included in the set.
    """
    weight = None
    lower_bound = normalizer.meta_feature_distributions[k].min_stat.get()
    scaling_factor = normalizer.meta_feature_distributions[k].max_stat.get() - lower_bound

    # If we only have seen one concept, we cannot work out how features change
    # across concepts. Our only data is the variance of each feature.
    # We scale each dimension such that 1 standard deviation is one unit.
    # (With a minimum stdev of 0.01)
    # Default values set to np.nan for 0 standard deviation.
    # These are replaced with the min seen weight.
    scale_factor = np.nan

    # The scale factor describes the range of concept values.
    # We cannot weight a single value, so return a default
    # np.nan is processed out later
    if scaling_factor <= 0:
        return np.nan

    standard_deviations = metafeature_stdevs_matrix[:, :, k][metafeature_initialized_matrix].tolist()
    overall_stdev = normalizer.meta_feature_distributions[k].stdev
    overall_mean = normalizer.meta_feature_distributions[k].mean

    if len(standard_deviations) == 0:
        return np.nan

    counts = metafeature_counts_matrix[:, :, k][metafeature_initialized_matrix].tolist()
    mean_intraconcept_stdev = 0
    for count, stdev in zip(counts, standard_deviations):
        mean_intraconcept_stdev += count * (stdev / scaling_factor)
    mean_intraconcept_stdev /= np.sum(counts)

    # The scale factor scales each dimension so
    # that a deviance in each dimension is relatively
    # equivalent.
    # This is based on the average intra concept stdev,
    # i.e. the normal variance we would expect to see within
    # a fingerprint.
    # 1/stdev scales the dimension such that the unit distance
    # is equal to one standard deviation, i.e. deviations
    # used to calculate similarity are in terms of standard
    # deviation.
    # To constrain this weight to the range (0, 1], we
    # clamp stdev to [0.01, inf) and multiply by 0.01.
    # This transformation considers all stdevs under 0.01
    # to be the same 'small' value.
    # This ensures unstable features which sometimes stay
    # very similar then change a lot, e.g. FI don't get
    # huge weights from small stdevs.
    # The * 0.01 means we consider
    # the unit distance to be 100 standard deviations.
    # The base meaning is the same under this transformation.
    scale_factor = 0.01 / max(mean_intraconcept_stdev, 0.01)

    scaled_overall_stdev = (overall_stdev) / scaling_factor

    scaled_overall_mean = (overall_mean - lower_bound) / scaling_factor

    if scaled_overall_stdev < 0:
        return np.nan

    # If the number of fingerprints is 1, we have not seen any concept drift.
    # We cannot identify which features will change, so just weight according to
    # standard deviation.
    if metafeature_initialized_matrix.sum() < 2:
        return scale_factor

    weight = weight_func(
        k,
        repository,
        normalizer,
        mi_formula,
        weighted,
        metafeature_matrix,
        metafeature_stdevs_matrix,
        metafeature_counts_matrix,
        metafeature_initialized_matrix,
        lower_bound,
        scaling_factor,
        scale_factor,
        scaled_overall_stdev,
        scaled_overall_mean,
    )
    return weight


def fisher_overall_weight(
    k: int,
    repository: Repository,
    normalizer: MetaFeatureNormalizer,
    mi_formula: Optional[Callable],
    weighted: bool,
    metafeature_matrix: np.ndarray,
    metafeature_stdevs_matrix: np.ndarray,
    metafeature_counts_matrix: np.ndarray,
    metafeature_initialized_matrix: np.ndarray,
    lower_bound: float,
    scaling_factor: float,
    scale_factor: float,
    scaled_overall_stdev: float,
    scaled_overall_mean: float,
) -> float:
    # Feature importance is made up of 2 factors.
    # Using the classifier for concept C, distinguish active and non-active fingerprints.
    # And given the classifier and fingerprint from a set of concepts determine the active one.

    n_states = metafeature_matrix.shape[0]
    active_state_mask = np.zeros((n_states, n_states), dtype=int)
    for i in range(n_states):
        active_state_mask[i, i] = 1

    # We first calculate the fisher score between active-nonactive for all concepts,
    # and get the average.

    between_segment_fisher_scores = []
    non_active_means = metafeature_matrix[:, :, k][np.logical_not(active_state_mask)].reshape(-1)
    non_active_scaled_means = (non_active_means - lower_bound) / scaling_factor
    non_active_counts = metafeature_counts_matrix[:, :, k][np.logical_not(active_state_mask)].reshape(-1)
    between_segment_fisher_score = fisher_score(non_active_scaled_means, non_active_counts, scaled_overall_stdev)
    between_segment_fisher_scores.append(between_segment_fisher_score)

    means = metafeature_matrix[:, :, k][active_state_mask].reshape(-1)
    scaled_means = (means - lower_bound) / scaling_factor
    counts = metafeature_counts_matrix[:, :, k][active_state_mask].reshape(-1)
    between_active_fisher_score = fisher_score(scaled_means, counts, scaled_overall_stdev)

    feature_importance_weight = max(between_active_fisher_score, float(np.mean(between_segment_fisher_scores)))
    weight = feature_importance_weight * scale_factor
    return weight
