""" Base state class"""
from __future__ import annotations

from river import utils
from river.base import Classifier
from river.base.typing import ClfTarget

from streamselect.concept_representations import (
    ConceptRepresentation,
    ErrorRateRepresentation,
    RepresentationComparer,
)


class State:  # pylint: disable=too-few-public-methods
    """A base state containing a Classifier and ConceptRepresentation"""

    def __init__(
        self,
        classifier: Classifier,
        concept_representation: ConceptRepresentation | None = None,
        state_id: int = -1,
        train_representation: bool = True,
    ) -> None:
        self.state_id = state_id
        self.classifier = classifier
        self.concept_representation = (
            concept_representation if concept_representation else ErrorRateRepresentation(window_size=1)
        )
        self.train_representation = train_representation

    def learn_one(self, x: dict, y: ClfTarget, sample_weight: float = 1.0) -> State:
        """Train the classifier and concept representation."""

        if self.train_representation:
            # Make a prediction without training statistics,
            # to avoid training twice.
            with utils.pure_inference_mode():
                p = self.classifier.predict_one(x)
            self.concept_representation.learn_one(x=x, y=y, p=p)

        # Some classifiers cannot take sample_weight.
        # Try/except to avoid branching
        try:
            self.classifier.learn_one(x=x, y=y, sample_weight=sample_weight)
        except TypeError:
            self.classifier.learn_one(x=x, y=y)
        return self

    def predict_one(self, x: dict) -> ClfTarget:
        """Make a prediction using the state classifier.
        Also trains unsupervised components of the classifier and concept representation."""
        p = self.classifier.predict_one(x)
        if self.train_representation:
            self.concept_representation.predict_one(x=x, p=p)
        return p

    def get_similarity_to_state(self, state_b: State, comparison: RepresentationComparer) -> float:
        """Return a similarity value between this state and another state."""
        return self.get_similarity_to_representation(state_b.concept_representation, comparison)

    def get_similarity_to_representation(
        self, rep_b: ConceptRepresentation, comparison: RepresentationComparer
    ) -> float:
        """Return a similarity value between this state and a concept representation."""
        return comparison.get_similarity(self.concept_representation, rep_b)

    def deactivate_train_representation(self) -> None:
        """Deactivate training representation.
        Some representations are not trained, e.g., implied error rate."""
        self.train_representation = False
