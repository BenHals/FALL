""" Classes to compare concept representations.
Classes are required to maintain learnable weights over time."""
import abc

from river.base import Base

from streamselect.states import State

from .base import ConceptRepresentation


class RepresentationComparer(Base, abc.ABC):
    """A base class able to compare concept representations.
    Different comparison strategies are available depending on representation.
    May maintain a set of weights to use during comparison."""

    @abc.abstractmethod
    def get_similarity(self, rep_a: ConceptRepresentation, rep_b: ConceptRepresentation) -> float:
        """Teturns the similarity between concept representations."""

    def get_state_similarity(self, state_a: State, state_b: State) -> float:
        """Returns the similarity between concept representations of the given states."""
        return self.get_similarity(state_a.concept_representation, state_b.concept_representation)
