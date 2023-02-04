import numpy as np

from fall.concept_representations import MetaFeatureNormalizer  # ConceptRepresentation
from fall.repository import Repository

# from fall.states import State


def uniform_weighting(repository: Repository, normalizer: MetaFeatureNormalizer) -> list[float]:
    n_metafeatures = 1
    for _, state in repository.states.items():
        active_representation = state.get_self_representation()
        if active_representation is None or not active_representation.initialized:
            continue
        n_metafeatures = len(active_representation.meta_feature_values)

    if n_metafeatures == 0:
        print("Warning: Metafeature dimensions cannot be determined")

    return [1.0] * n_metafeatures


def random_weighting(repository: Repository, normalizer: MetaFeatureNormalizer) -> list[float]:
    n_metafeatures = 1
    for _, state in repository.states.items():
        active_representation = state.get_self_representation()
        if active_representation is None or not active_representation.initialized:
            continue
        n_metafeatures = len(active_representation.meta_feature_values)

    if n_metafeatures == 0:
        print("Warning: Metafeature dimensions cannot be determined")

    return np.random.rand(n_metafeatures).tolist()
