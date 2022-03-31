""" Base adaptive learning class. """
import abc
from typing import Callable, Dict, Tuple

from river.base import Classifier, DriftDetector
from river.base.typing import ClfTarget
from river.utils import pure_inference_mode

from streamselect.adaptive_learning.buffer import SupervisedUnsupervisedBuffer
from streamselect.concept_representations import ConceptRepresentation
from streamselect.repository import Repository, RepresentationComparer, ValuationPolicy
from streamselect.states import State


class BaseAdaptiveLearner(Classifier, abc.ABC):
    """A base adaptive learning class."""

    def __init__(
        self,
        classifier_constructor: Callable[[], Classifier],
        representation_constructor: Callable[[int], ConceptRepresentation],
        representation_comparer: RepresentationComparer,
        drift_detector: DriftDetector,
        max_size: int = -1,
        valuation_policy: ValuationPolicy = ValuationPolicy.NoPolicy,
        train_representation: bool = True,
        window_size: int = 1,
        construct_pair_representations: bool = False,
    ):
        """
        Parameters
        ----------
        classifier_constructor: Callable[[], Classifier]
            A function to generate a new classifier.

        representation_constructor: Callable[[], ConceptRepresentation]
            A function to generate a new concept representation, or none.

        representation_comparer: RepresentationComparer
            An object capable of calculating similarity between two representations.

        drift_detector: DriftDetector
            An object capable of detecting drift in a univariate stream.

        max_size: int
            Default -1
            Maximum number of states able to be stored.
            -1 represents unlimited. Anything else requires
            a deletion valuation policy.

        valuation_policy: ValiationPolicy
        Default NoPolicy
            The valuation policy to use. Must be set if max_size != -1

        train_representation: bool
            Whether or not new states train representations.
            Must be set to automatically construct states.

        window_size: int
            The number of observations to construct a concept representation over.

        construct_pair_representations: bool
            Whether or not to construct a representation for each concept, classifier pair.
            Such a pair R_{ij} represents data drawn from concept j classified by state i.
            If False, only constructs R_{ii}, which is fine for many adaptive learning systems.

        """
        self.max_size = max_size
        self.valuation_policy = valuation_policy
        self.classifier_constructor = classifier_constructor
        self.representation_constructor = lambda: representation_constructor(window_size)
        self.representation_comparer = representation_comparer
        self.drift_detector = drift_detector
        self.train_representation = train_representation
        self.window_size = window_size
        self.construct_pair_representations = construct_pair_representations

        # timestep for unsupervised data
        self.supervised_timestep = 0
        # timestep for supervised data
        self.unsupervised_timestep = 0

        self.active_state_id: int = 0
        self.repository = Repository(
            max_size=self.max_size,
            valuation_policy=self.valuation_policy,
            classifier_constructor=self.classifier_constructor,
            representation_constructor=self.representation_constructor,
            train_representation=self.train_representation,
        )
        self.repository.add_next_state()

        self.recent_representation = self.representation_constructor()

    def predict_one(self, x: dict) -> ClfTarget:
        """Make a prediction using the active state classifier.
        Also trains unsupervised components of the classifier and concept representation.
        """
        active_state = self.get_active_state()
        with pure_inference_mode():
            p = active_state.predict_one(x, self.active_state_id)
        self.recent_representation.predict_one(x, p)

        # Train unsupervised representation features
        trained_states = self.repository.states.values() if self.construct_pair_representations else [active_state]
        for state in trained_states:
            state.predict_one(x, self.active_state_id)

        self.representation_comparer.train_unsupervised(self.repository)
        self.unsupervised_timestep += 1
        return p

    def learn_one(self, x: dict, y: ClfTarget, sample_weight: float = 1.0) -> None:
        active_state = self.get_active_state()

        # train supervised representation features and state classifier.
        trained_states = self.repository.states.values() if self.construct_pair_representations else [active_state]
        for state in trained_states:
            state.learn_one(
                x=x,
                y=y,
                concept_id=self.active_state_id,
                sample_weight=sample_weight,
            )

        # Train recent concept representation
        with pure_inference_mode():
            p = active_state.predict_one(x, self.active_state_id)
        self.recent_representation.learn_one(x, y, p)

        self.supervised_timestep += 1
        self.step()

    def step(self) -> None:
        """Update internal state"""
        # Update state statistics
        self.repository.step_all(self.active_state_id)

        self.representation_comparer.train_supervised(self.repository)

        in_drift, in_warning, _ = self.perform_drift_detection()

        if in_drift:
            state_relevance = self.perform_reidentification()
            new_active_state = self.get_adapted_state(state_relevance)
            self.transition_active_state(new_active_state, True, in_warning)

    def get_active_state(self) -> State:
        """Return the currently active state."""
        return self.repository.states[self.active_state_id]

    def perform_drift_detection(self) -> Tuple[bool, bool, float]:
        """Monitors the relevance of the currently active state.
        returns whether a drift or warning has been detected, and the relevance.

        Returns
        -------
        in_drift: bool
            True if a drift was detected.

        in_warning: bool
            True if a warning was detected

        active_relevance: float
            The relevance of the active state to recent data."""

        active_state = self.get_active_state()
        active_state_relevance = self.representation_comparer.get_state_rep_similarity(
            active_state, self.recent_representation
        )

        in_drift, in_warning = self.drift_detector.update(active_state_relevance)  # type: ignore

        return in_drift, in_warning, active_state_relevance

    def perform_reidentification(self) -> Dict[int, float]:
        """Estimate the relevance of each state in the repository to current data."""
        return {k: 0.0 for k in self.repository.states}

    def get_adapted_state(self, state_relevance: Dict[int, float]) -> State:
        """Returns a new state adapted to current conditions, based on estimated relevance
        of previous states."""

        new_state = self.repository.add_next_state()
        return new_state

    def transition_active_state(self, next_active_state: State, in_drift: bool, in_warning: bool) -> None:
        """Transition to a new active state."""
        current_active_state = self.get_active_state()
        self.active_state_id = next_active_state.state_id
        self.repository.add_transition(
            current_active_state, next_active_state, in_drift=in_drift, in_warning=in_warning
        )


class BaseBufferedAdaptiveLearner(BaseAdaptiveLearner):
    """A base adaptive learning class, using a buffer to stabilize observations."""

    def __init__(
        self,
        classifier_constructor: Callable[[], Classifier],
        representation_constructor: Callable[[int], ConceptRepresentation],
        representation_comparer: RepresentationComparer,
        drift_detector: DriftDetector,
        max_size: int = -1,
        valuation_policy: ValuationPolicy = ValuationPolicy.NoPolicy,
        train_representation: bool = True,
        window_size: int = 1,
        buffer_timeout: float = 0.0,
        construct_pair_representations: bool = False,
    ):
        """
        Parameters
        ----------
        classifier_constructor: Callable[[], Classifier]
            A function to generate a new classifier.

        representation_constructor: Callable[[], ConceptRepresentation]
            A function to generate a new concept representation, or none.

        representation_comparer: RepresentationComparer
            An object capable of calculating similarity between two representations.

        drift_detector: DriftDetector
            An object capable of detecting drift in a univariate stream.

        max_size: int
            Default -1
            Maximum number of states able to be stored.
            -1 represents unlimited. Anything else requires
            a deletion valuation policy.

        valuation_policy: ValiationPolicy
        Default NoPolicy
            The valuation policy to use. Must be set if max_size != -1

        train_representation: bool
            Whether or not new states train representations.
            Must be set to automatically construct states.

        window_size: int
            The number of observations to construct a concept representation over.

        buffer_timeout: float
            The base number of timesteps to buffer new data before training.

        construct_pair_representations: bool
            Whether or not to construct a representation for each concept, classifier pair.
            Such a pair R_{ij} represents data drawn from concept j classified by state i.
            If False, only constructs R_{ii}, which is fine for many adaptive learning systems.

        """
        super().__init__(
            classifier_constructor,
            representation_constructor,
            representation_comparer,
            drift_detector,
            max_size,
            valuation_policy,
            train_representation,
            window_size,
            construct_pair_representations,
        )
        self.buffer_timeout = buffer_timeout

        # Create a buffer to hold supervised and unsupervised data.
        # Uses a clock based on supervised data, i.e., supervised data is considered stable when it is older
        # than buffer_timout supervised observations, and all unsupervised observations collected
        # before this observation is also released.
        self.buffer = SupervisedUnsupervisedBuffer(
            self.window_size, self.buffer_timeout, self.buffer_timeout, release_strategy="supervised"
        )

    def predict_one(self, x: dict) -> ClfTarget:
        """Make a prediction using the active state classifier.
        Also trains unsupervised components of the classifier and concept representation.

        Note: A buffer is used to only train on stable"""
        active_state = self.get_active_state()
        with pure_inference_mode():
            p = active_state.predict_one(x, self.active_state_id)
        self.recent_representation.predict_one(x, p)

        # Train unsupervised representation features
        # In the buffered version, we train on observations coming out
        # of the buffer, rather than new observations.
        self.buffer.buffer_unsupervised(x)
        stable_data = self.buffer.collect_stable_unsupervised()
        for stable_observation in stable_data:
            trained_states = self.repository.states.values() if self.construct_pair_representations else [active_state]
            for state in trained_states:
                state.predict_one(stable_observation.x, self.active_state_id)

        self.representation_comparer.train_unsupervised(self.repository)
        return p

    def learn_one(self, x: dict, y: ClfTarget, sample_weight: float = 1.0) -> None:
        active_state = self.get_active_state()
        self.buffer.buffer_supervised(x, y)
        stable_data = self.buffer.collect_stable_supervised()

        # train supervised representation features and state classifier.
        # In the buffered version, we train on observations coming out
        # of the buffer, rather than new observations.
        for stable_observation in stable_data:
            if stable_observation.y is None:
                continue
            trained_states = self.repository.states.values() if self.construct_pair_representations else [active_state]
            for state in trained_states:
                state.learn_one(
                    x=stable_observation.x,
                    y=stable_observation.y,
                    concept_id=self.active_state_id,
                    sample_weight=stable_observation.sample_weight,
                )

        # Train recent concept representation
        with pure_inference_mode():
            p = active_state.predict_one(x, self.active_state_id)
        self.recent_representation.learn_one(x, y, p)

        self.step()

    def transition_active_state(self, next_active_state: State, in_drift: bool, in_warning: bool) -> None:
        super().transition_active_state(next_active_state, in_drift, in_warning)
        self.buffer.reset_on_drift()