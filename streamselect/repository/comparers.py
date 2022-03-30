""" Classes to compare concept representations.
Classes are required to maintain learnable weights over time."""
import abc
from typing import List

from river.base import Base

from streamselect.concept_representations import ConceptRepresentation
from streamselect.repository import Repository
from streamselect.states import State


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
        return 1 - abs(rep_a.values[0] - rep_b.values[0])
