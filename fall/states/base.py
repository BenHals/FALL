""" Base state class"""
from __future__ import annotations

from typing import Callable

from river.base import Classifier
from river.base.typing import ClfTarget
from river.compose import pure_inference_mode
from river.drift import ADWIN

from fall.concept_representations import (
    BaseDistribution,
    ConceptRepresentation,
    GaussianDistribution,
)
from fall.utils import Observation


class State:  # pylint: disable=too-few-public-methods
    """A base state containing a Classifier and ConceptRepresentation.

    Parameters

    classifier: Classifier
        - The base classifier to use for prediction.

    representation_constructor: Callable[[int], ConceptRepresentation]
        - A function used to generate a new concept representation

    state_id: int
        - The id of the state. Should be positive and unique
        (The ID -1 is reserved for the background state)

    train_representation: bool
        - A flag for enabling or disabling training the concept representation.
        Some concept representations may not be trained, e.g., the error rate.

    in_concept_relevance_memory_size: int
        - How many observations of the relevance recorded within the associated concept
        should be stored. Higher may give a more accurate distribution, but we may also
        want to drop early records which are no longer comparable due to changes to weighting

    """

    def __init__(
        self,
        classifier: Classifier,
        representation_constructor: Callable[[int], ConceptRepresentation],
        state_id: int = -1,
        train_representation: bool = True,
        in_concept_relevance_memory_size: int = 25,
    ) -> None:
        self.state_id = state_id
        self.name = str(self.state_id)
        self.classifier = classifier
        self.representation_constructor = representation_constructor
        # Mapping between concept ids and representations using  self.classifier.
        self.concept_representation: dict[int, ConceptRepresentation] = {
            self.state_id: self.representation_constructor(self.state_id)
        }
        self.train_representation = train_representation
        self.in_concept_relevance_memory_size = in_concept_relevance_memory_size

        self.seen_weight = 0.0
        self.active_seen_weight = 0.0
        self.seen_weight_since_active = 0.0
        self.weight_since_last_active = 0.0
        self.last_trained_active_timestep = -1.0

        self.evolved_at_last_update = False
        self.weight_since_last_evolution = 0.0

        # We store a record of this states accuracy while active
        # (i.e., on observations with an active_state_id set to this states id)
        # We can use ADWIN for this, to keep the most recent window of
        # accuracy measurements with the same mean.
        self.in_concept_accuracy_record = ADWIN()

        # We track the states relevance while active in the same way, and also record a distribution.
        self.in_concept_relevance_record = ADWIN()
        self.in_concept_relevance_distribution: BaseDistribution = GaussianDistribution(
            memory_size=self.in_concept_relevance_memory_size
        )

        # We finally track a recent relevance.
        self.current_relevance_record = ADWIN()

    def learn_one(self, supervised_observation: Observation, force_train_classifier: bool = False) -> State:
        """Train the classifier and concept representation.
        concept_id determines the concept the observation is thought to be drawn from.
        The state classifier is NOT trained on observations with a concept_id which does not match
        the state_id unless force_train_classifier is set, however other statistics are updated.

        Parameters
        ----------

        supervised_observation: Observation
            The observation to train on. Must be supervised, i.e., have a valid y value.
            The prediction on the observation is not used, but a new one is added to ensure
            that the most up to date predictions are used.

        force_train_classifier: bool
            Default: False
            Forces the state classifier to train on an observation regardless of which concept_id
            the observation is from. When false, only observations with with an active_state_id
            matching the state_id are used to train the classifier.

        """
        concept_id = supervised_observation.active_state_id if not force_train_classifier else self.state_id
        if self.train_representation:
            representation = self.concept_representation.setdefault(
                concept_id, self.representation_constructor(self.state_id)
            )
            # Make a prediction without training statistics,
            # to avoid training twice.
            with pure_inference_mode():
                p = self.classifier.predict_one(supervised_observation.x)
                supervised_observation.add_prediction(p, self.state_id)
            representation.learn_one(supervised_observation)

        # We only train the classifier on data from the associated concept.
        if concept_id != self.state_id:
            return self

        # We only train the classifier on data from the associated concept.
        if supervised_observation.y is None:
            raise ValueError("Attempting to train on unsupervised observation. ")

        # Some classifiers cannot take sample_weight.
        # Try/except to avoid branching
        prior_evolution = self.get_current_evolution()
        try:
            self.classifier.learn_one(
                x=supervised_observation.x,
                y=supervised_observation.y,
                sample_weight=supervised_observation.sample_weight,
            )
        except TypeError:
            self.classifier.learn_one(x=supervised_observation.x, y=supervised_observation.y)
        post_evolution = self.get_current_evolution()
        self.evolved_at_last_update = False
        if post_evolution != prior_evolution:
            self.evolved_at_last_update = True
            self.weight_since_last_evolution = 0.0

        is_correct = supervised_observation.y == supervised_observation.predictions[self.state_id]
        self.in_concept_accuracy_record.update(int(is_correct))  # type: ignore

        if supervised_observation.active_state_relevance is not None and self.get_self_representation().stable:
            self.add_active_state_relevance(supervised_observation.active_state_relevance)

        self.last_trained_active_timestep = supervised_observation.seen_at
        return self

    def predict_one(
        self, unsupervised_observation: Observation, force_train_own_representation: bool = False
    ) -> ClfTarget:
        """Make a prediction using the state classifier.
        Also trains unsupervised components of the classifier and concept representation.

        Parameters
        ----------

        unsupervised_observation: Observation
            An unsupervised observation, may have a None y.
            Predictions for the state_id will be added to observation.predictions.

        force_train_own_representation: Bool
            Default: False
            Forces the state to train the concept_representation representing data with the current state_id.
            If false, we train the representation representing the active_state_id associated with the observation.
        """
        p = self.classifier.predict_one(unsupervised_observation.x)
        unsupervised_observation.add_prediction(p, self.state_id)
        if self.train_representation:
            concept_id = (
                unsupervised_observation.active_state_id if not force_train_own_representation else self.state_id
            )
            representation = self.concept_representation.setdefault(
                concept_id, self.representation_constructor(self.state_id)
            )
            representation.predict_one(unsupervised_observation)
        return p

    def get_current_evolution(self) -> int:
        """Get the current evolution id of the classifier.
        Returns 0 if the classifier does not track evolutions."""
        if not hasattr(self.classifier, "evolutions"):
            return 0
        return self.classifier.evolutions

    def step(self, sample_weight: float = 1.0, is_active: bool = True) -> None:
        """Step states tracking statistics"""
        self.seen_weight += sample_weight
        self.weight_since_last_active += sample_weight
        if is_active:
            self.active_seen_weight += sample_weight
            self.seen_weight_since_active += sample_weight
            self.weight_since_last_evolution += sample_weight
            self.weight_since_last_active = 0

    def add_active_state_relevance(self, active_state_relevance: float) -> None:
        """Update statistics tracking active_state_relevance.
        Should only be called when an observation is deemed stable,
        and when the current representation is stable (has seen enough)
        elements to calculate accurate statistics."""

        self.in_concept_relevance_distribution.learn_one(active_state_relevance)
        self.in_concept_relevance_record.update(active_state_relevance)
        self.current_relevance_record.update(active_state_relevance)

    def get_in_concept_relevance(self) -> float:
        return self.in_concept_relevance_record.estimation

    def get_current_relevance(self) -> float:
        return self.current_relevance_record.estimation

    def get_self_representation(self) -> ConceptRepresentation:
        """Get the concept representation using this states classifier,
        on data drawn from this concept."""
        return self.concept_representation[self.state_id]

    def deactivate_train_representation(self) -> None:
        """Deactivate training representation.
        Some representations are not trained, e.g., implied error rate."""
        self.train_representation = False

    def transition_from(self) -> None:
        self.seen_weight_since_active = 0.0

    def __str__(self) -> str:
        return f"<State {self.state_id}>"

    def __repr__(self) -> str:
        return str(self)
