""" Classes to compare concept representations.
Classes are required to maintain learnable weights over time."""
import abc
from typing import List

import numpy as np
from river.base import Base
from scipy.spatial.distance import cosine

from fall.concept_representations import ConceptRepresentation
from fall.repository import Repository
from fall.states import State


class RepresentationComparer(Base, abc.ABC):
    """A base class able to compare concept representations.
    Different comparison strategies are available depending on representation.
    May maintain a set of weights to use during comparison."""

    @abc.abstractmethod
    def get_similarity(self, rep_a: ConceptRepresentation, rep_b: ConceptRepresentation) -> float:
        """Returns the similarity between concept representations."""

    def get_state_similarity(self, state_a: State, state_b: State) -> float:
        """Returns the similarity between states."""
        return self.get_similarity(state_a.get_self_representation(), state_b.get_self_representation())

    def get_state_rep_similarity(self, state_a: State, rep_b: ConceptRepresentation) -> float:
        """Returns the similarity between a state and a concept representations."""
        return self.get_similarity(state_a.get_self_representation(), rep_b)

    def train_supervised(self, repository: Repository) -> None:
        """Train trainable components on the repository."""

    def train_unsupervised(self, repository: Repository) -> None:
        """Train trainable components on the repository."""


class AbsoluteValueComparer(RepresentationComparer):
    """A representation comparer which calculates similarity
    based on the absolute value difference in value between representations.
    Where multiple meta-features are present, only the first is compared.

    We assume that values are in the range [0, 1].
    Thus to return similarity, we return 1 - abs(difference).
    E.G: an absolute difference of 0 is a similarity of 1,
    while a difference of 1 is a similarity of 0."""

    def __init__(self) -> None:
        self.weights: List[float] = [1.0]

    def get_similarity(self, rep_a: ConceptRepresentation, rep_b: ConceptRepresentation) -> float:
        return 1 - abs(rep_a.meta_feature_values[0] - rep_b.meta_feature_values[0])


class CosineComparer(RepresentationComparer):
    """A representation comparer which calculates similarity
    based on the cosine distance between meta-feature vectors.

    To return similarity, we return 1 - abs(cosine_distance).
    E.G: an absolute difference of 0 is a similarity of 1,
    while a difference of 1 is a similarity of 0."""

    def __init__(self) -> None:
        self.weights: List[float] = [1.0]
        self.initialized = False

    def initialize(self, vec: list[float]) -> None:
        self.weights = [1.0] * len(vec)
        self.initialized = True

    def get_similarity(self, rep_a: ConceptRepresentation, rep_b: ConceptRepresentation) -> float:
        if not self.initialized:
            self.initialize(rep_a.meta_feature_values)

        # Normalize the vectors, using the overall normalizer
        # i.e., the global distribution of the meta-feature.
        # We use the global, because two distinct concepts may locally normalize
        # different vectors to the same value.
        vec_a = np.array(rep_a.overall_normalize(rep_a.meta_feature_values))
        vec_b = np.array(rep_b.overall_normalize(rep_b.meta_feature_values))
        weights = np.array(self.weights)
        print(vec_a, vec_b, get_cosine_distance(vec_a, vec_b, weights))
        return 1 - get_cosine_distance(vec_a, vec_b, weights)


def get_cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray, weights: np.ndarray) -> float:
    """Get the weighted cosine distance between two vectors.
    Internally normalizes weights.
    """
    normed_weights = (weights) / (np.max(weights))
    try:
        c = cosine(vec_a, vec_b, w=normed_weights)
    except ZeroDivisionError:
        c = np.nan
    if np.isnan(c):
        c = 0 if ((not np.any(vec_a)) and (not np.any(vec_b))) else 1
    return c
