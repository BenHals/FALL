""" Base state class"""
from __future__ import annotations

from typing import Callable

from river import utils
from river.base import Classifier
from river.base.typing import ClfTarget

from streamselect.concept_representations import ConceptRepresentation


class State:  # pylint: disable=too-few-public-methods
    """A base state containing a Classifier and ConceptRepresentation"""

    def __init__(
        self,
        classifier: Classifier,
        representation_constructor: Callable[[], ConceptRepresentation],
        state_id: int = -1,
        train_representation: bool = True,
    ) -> None:
        self.state_id = state_id
        self.classifier = classifier
        self.representation_constructor = representation_constructor
        # Mapping between concept ids and representations using  self.classifier.
        self.concept_representation: dict[int, ConceptRepresentation] = {
            self.state_id: self.representation_constructor()
        }
        self.train_representation = train_representation

        self.seen_weight = 0.0
        self.active_seen_weight = 0.0
        self.weight_since_last_active = 0.0

    def learn_one(self, x: dict, y: ClfTarget, concept_id: int | None = None, sample_weight: float = 1.0) -> State:
        """Train the classifier and concept representation.
        concept_id determines the concept the observation is thought to be drawn from.
        The state classifier is NOT trained on observations with a concept_id which does not match
        the state_id, however other statistics are updated."""
        if concept_id is None:
            concept_id = self.state_id
        if self.train_representation:
            representation = self.concept_representation.setdefault(concept_id, self.representation_constructor())
            # Make a prediction without training statistics,
            # to avoid training twice.
            with utils.pure_inference_mode():
                p = self.classifier.predict_one(x)
            representation.learn_one(x=x, y=y, p=p)

        # We only train the classifier on data from the associated concept.
        if concept_id != self.state_id:
            return self

        # Some classifiers cannot take sample_weight.
        # Try/except to avoid branching
        try:
            self.classifier.learn_one(x=x, y=y, sample_weight=sample_weight)
        except TypeError:
            self.classifier.learn_one(x=x, y=y)

        return self

    def predict_one(self, x: dict, concept_id: int | None = None) -> ClfTarget:
        """Make a prediction using the state classifier.
        Also trains unsupervised components of the classifier and concept representation."""
        if concept_id is None:
            concept_id = self.state_id
        p = self.classifier.predict_one(x)
        if self.train_representation:
            representation = self.concept_representation.setdefault(concept_id, self.representation_constructor())
            representation.predict_one(x=x, p=p)
        return p

    def step(self, sample_weight: float = 1.0, is_active: bool = True) -> None:
        """Step states tracking statistics"""
        self.seen_weight += sample_weight
        self.weight_since_last_active += sample_weight
        if is_active:
            self.active_seen_weight += sample_weight
            self.weight_since_last_active = 0

    def get_self_representation(self) -> ConceptRepresentation:
        """Get the concept representation using this states classifier,
        on data drawn from this concept."""
        return self.concept_representation[self.state_id]

    def deactivate_train_representation(self) -> None:
        """Deactivate training representation.
        Some representations are not trained, e.g., implied error rate."""
        self.train_representation = False
