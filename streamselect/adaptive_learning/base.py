""" Base adaptive learning class. """
import abc
from collections import deque
from typing import Callable, Deque, Dict, Optional, Set, Tuple, Union

from river.base import Classifier, DriftDetector
from river.base.typing import ClfTarget
from river.utils import pure_inference_mode

from streamselect.adaptive_learning.buffer import (
    Observation,
    SupervisedUnsupervisedBuffer,
)
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
        drift_detector_constructor: Callable[[], DriftDetector],
        max_size: int = -1,
        valuation_policy: ValuationPolicy = ValuationPolicy.NoPolicy,
        train_representation: bool = True,
        window_size: int = 1,
        construct_pair_representations: bool = False,
        prediction_mode: str = "active",
        background_state_mode: Union[str, int, None] = "drift_reset",
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

        drift_detector_constructor: Callable[[], DriftDetector]
            A function to generate an object capable of detecting drift in a univariate stream.

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
            Default: 1
            The number of observations to construct a concept representation over.

        construct_pair_representations: bool
            Default: False
            Whether or not to construct a representation for each concept, classifier pair.
            Such a pair R_{ij} represents data drawn from concept j classified by state i.
            If False, only constructs R_{ii}, which is fine for many adaptive learning systems.

        prediction_mode: str ["active", "all"]
            Default: "active"
            Which states must make predictions.
            "active" only makes predictions with the active state
            "all" makes predictions with all classifiers, e.g., for an ensemble.

        background_state_mode: str["drift_reset", "transition_reset"] | int | None
            Default: "drift_reset"
            Mode for background state. If None, no background state is run.
            Otherwise, a background state is created and run alongside the active state
            to determine when transitioning to a new state is optimal. The mode determines
            how the background state is reset to match with recent data.
            drift_reset: The state is reset when a drift in it's representation is detected
            transition_reset: The state is reset only when a transition occurs
            int>0: timed_reset, the state is reset periodically every int timesteps.

        """
        self.max_size = max_size
        self.valuation_policy = valuation_policy
        self.classifier_constructor = classifier_constructor
        self.representation_constructor = lambda: representation_constructor(window_size)
        self.representation_comparer = representation_comparer
        self.drift_detector_constructor = drift_detector_constructor
        self.train_representation = train_representation
        self.window_size = window_size
        self.construct_pair_representations = construct_pair_representations
        self.prediction_mode = prediction_mode
        self.background_state_mode = background_state_mode

        # Validation
        if self.prediction_mode != "all" and self.construct_pair_representations:
            raise ValueError(
                "Prediction mode not set to all, but construct_pair_representation requires all predictions."
            )

        # timestep for unsupervised data
        self.supervised_timestep = 0
        # timestep for supervised data
        self.unsupervised_timestep = 0

        self.unsupervised_active_window: Deque[Observation] = deque(maxlen=self.window_size)
        self.supervised_active_window: Deque[Observation] = deque(maxlen=self.window_size)

        self.repository = Repository(
            max_size=self.max_size,
            valuation_policy=self.valuation_policy,
            classifier_constructor=self.classifier_constructor,
            representation_constructor=self.representation_constructor,
            train_representation=self.train_representation,
        )
        self.drift_detector = self.drift_detector_constructor()
        active_state: State = self.repository.add_next_state()
        self.active_state_id: int = active_state.state_id

        self.active_window_state_representations: Dict[int, ConceptRepresentation] = {
            self.active_state_id: self.representation_constructor()
        }

        self.setup_background_state()

    def setup_background_state(self) -> None:
        """Setup or reset the background state."""
        self.background_state: Optional[State] = None
        self.background_state_active_representation: Optional[ConceptRepresentation] = None
        self.background_state_detector: Optional[DriftDetector] = None
        if self.background_state_mode:
            self.background_state = self.repository.make_state(-1)
            self.background_state_active_representation = self.representation_constructor()
            if self.background_state_mode == "drift_reset":
                self.background_state_detector = self.drift_detector_constructor()

    def predict_one(self, x: dict) -> ClfTarget:
        """Make a prediction using the active state classifier.
        Also trains unsupervised components of the classifier and concept representation.
        """
        state_predictions = self.repository.get_repository_predictions(x, self.active_state_id, self.prediction_mode)
        p = self.combine_predictions(state_predictions)

        self.train_components_unsupervised(x, state_predictions)

        return p

    def combine_predictions(self, state_predictions: Dict[int, ClfTarget]) -> ClfTarget:
        """Combines state predictions into a single output prediction."""
        return state_predictions[self.active_state_id]

    def train_background_unsupervised(self, x: dict, state_predictions: Dict[int, ClfTarget]) -> None:
        if self.background_state:
            bp = self.background_state.predict_one(x, concept_id=None)
            if self.background_state_active_representation:
                self.background_state_active_representation.predict_one(x, bp)

    def train_components_unsupervised(self, x: dict, state_predictions: Dict[int, ClfTarget]) -> None:
        """Train non-state components with unsupervised data."""
        # Train representations of recent data
        for state_id, representation in self.active_window_state_representations.items():
            state_p = state_predictions[state_id]
            representation.predict_one(x, state_p)

        self.representation_comparer.train_unsupervised(self.repository)
        self.unsupervised_active_window.append(Observation(x, None, 1.0, self.unsupervised_timestep))

        self.unsupervised_timestep += 1

    def learn_one(self, x: dict, y: ClfTarget, sample_weight: float = 1.0) -> None:
        active_state = self.get_active_state()

        # train supervised representation features and state classifier.
        trainable_states = self.repository.states.values() if self.construct_pair_representations else [active_state]
        for state in trainable_states:
            state.learn_one(
                x=x,
                y=y,
                concept_id=self.active_state_id,
                sample_weight=sample_weight,
            )

        # Train recent concept representation for active state
        self.train_components_supervised(x, y, sample_weight)

        self.step()

    def train_background_supervised(self, x: dict, y: ClfTarget, sample_weight: float = 1.0) -> None:
        if self.background_state:
            with pure_inference_mode():
                bp = self.background_state.predict_one(x, concept_id=None)
                self.background_state.learn_one(x, y, concept_id=None, sample_weight=sample_weight)
            if self.background_state_active_representation:
                self.background_state_active_representation.learn_one(x, y, bp)
            self.background_state.step(sample_weight, is_active=True)

    def train_components_supervised(self, x: dict, y: ClfTarget, sample_weight: float = 1.0) -> None:
        """Train non-state components with supervised data."""
        with pure_inference_mode():
            state_predictions = self.repository.get_repository_predictions(
                x, self.active_state_id, self.prediction_mode
            )

        for state_id, representation in self.active_window_state_representations.items():
            state_p = state_predictions[state_id]
            representation.learn_one(x, y, state_p)

        self.representation_comparer.train_supervised(self.repository)
        self.supervised_active_window.append(Observation(x, y, sample_weight, self.supervised_timestep))
        self.supervised_timestep += 1

    def step(self) -> None:
        """Update internal state"""
        # Update state statistics
        self.repository.step_all(self.active_state_id)

        in_drift, in_warning, _ = self.active_state_drift_detection()

        self.evaluate_background_state(in_drift, in_warning)

        if in_drift:
            state_relevance = self.perform_reidentification()
            new_active_state = self.get_adapted_state(state_relevance)
            self.transition_active_state(new_active_state, True, in_warning)

    def evaluate_background_state(self, in_drift: bool, in_warning: bool) -> None:
        """Step the background state, and reset if needed."""
        if not self.background_state:
            return

        reset_required = False
        if self.background_state_mode == "drift_reset":
            if self.background_state_active_representation:
                b_in_drift, _, _ = self.perform_drift_detection(
                    self.background_state, self.background_state_active_representation
                )
                reset_required = b_in_drift
        elif self.background_state_mode == "transition_reset":
            reset_required = in_drift
        elif isinstance(self.background_state_mode, int):
            reset_required = self.background_state.seen_weight > self.background_state_mode

        if reset_required:
            self.setup_background_state()

    def get_active_state(self) -> State:
        """Return the currently active state."""
        return self.repository.states[self.active_state_id]

    def active_state_drift_detection(self) -> Tuple[bool, bool, float]:
        """Monitors the relevance of the active state.
        Returns
        -------
        in_drift: bool
            True if a drift was detected.

        in_warning: bool
            True if a warning was detected

        relevance: float
            The relevance of the state to recent data."""

        active_state = self.get_active_state()
        active_representation = self.active_window_state_representations[active_state.state_id]
        return self.perform_drift_detection(active_state, active_representation)

    def perform_drift_detection(
        self, state: State, state_representation: ConceptRepresentation
    ) -> Tuple[bool, bool, float]:
        """Monitors the relevance of a state.
        returns whether a drift or warning has been detected, and the relevance.

        Returns
        -------
        in_drift: bool
            True if a drift was detected.

        in_warning: bool
            True if a warning was detected

        relevance: float
            The relevance of the state to recent data."""

        state_relevance = self.representation_comparer.get_state_rep_similarity(state, state_representation)

        in_drift, in_warning = self.drift_detector.update(state_relevance)  # type: ignore

        return in_drift, in_warning, state_relevance

    def get_unsupervised_active_window(self) -> Deque[Observation]:
        """Returns the unsupervised active window as a deque of observations."""
        return self.unsupervised_active_window

    def get_supervised_active_window(self) -> Deque[Observation]:
        """Returns the supervised active window as a deque of observations."""
        return self.supervised_active_window

    def construct_active_representation(self, state: State, mode: str = "supervised") -> ConceptRepresentation:
        """Construct a new concept representation for the given state based on the current active window.

        Parameters
        ----------
        mode: str["supervised", "unsupervised", "both"]
            Default: "supervised"
            Whther to use supervised, unsupervised, or both, active windows to train the representation."""
        representation = self.representation_constructor()
        supervised_timesteps = set()
        if mode in ["supervised", "both"]:
            for observation in self.get_supervised_active_window():
                if observation.y is None:
                    continue
                supervised_timesteps.add(observation.seen_at)
                p = state.predict_one(observation.x)
                representation.predict_one(observation.x, p)
                representation.learn_one(observation.x, observation.y, p)

        if mode in ["unsupervised", "both"]:
            for observation in self.get_unsupervised_active_window():
                if observation.seen_at in supervised_timesteps:
                    continue
                p = state.predict_one(observation.x)
                representation.predict_one(observation.x, p)

        return representation

    def perform_reidentification(self) -> Dict[int, float]:
        """Estimate the relevance of each state in the repository to current data."""
        state_relevance: Dict[int, float] = {}
        for state_id, state in self.repository.states.items():
            active_representation = self.active_window_state_representations.get(
                state_id, self.construct_active_representation(state)
            )
            state_relevance[state_id] = self.representation_comparer.get_state_rep_similarity(
                state, active_representation
            )

        if self.background_state and self.background_state_active_representation:
            state_relevance[-1] = self.representation_comparer.get_state_rep_similarity(
                self.background_state, self.background_state_active_representation
            )
        return state_relevance

    def get_adapted_state(self, state_relevance: Dict[int, float]) -> State:
        """Returns a new state adapted to current conditions, based on estimated relevance
        of previous states."""
        max_state_id, _ = max(state_relevance.items(), key=lambda x: x[1])
        if max_state_id == -1 and self.background_state:
            new_state = self.repository.add_next_state()
            new_state.classifier = self.background_state.classifier
            new_state.concept_representation = self.background_state.concept_representation
        else:
            new_state = self.repository.states[max_state_id]
        return new_state

    def transition_active_state(self, next_active_state: State, in_drift: bool, in_warning: bool) -> None:
        """Transition to a new active state."""
        current_active_state = self.get_active_state()
        self.active_state_id = next_active_state.state_id
        self.repository.add_transition(
            current_active_state, next_active_state, in_drift=in_drift, in_warning=in_warning
        )

        self.transition_reset(current_active_state.state_id, next_active_state.state_id, in_drift, in_warning)

    def transition_reset(
        self, prev_active_state_id: int, next_active_state_id: int, in_drift: bool, in_warning: bool
    ) -> None:
        """Reset statistics after a transition."""

        # Create an active representation for the states we are predicting for.
        representation_ids: Set[int] = {next_active_state_id}
        if self.prediction_mode == "all":
            representation_ids = set(self.repository.states)

        self.active_window_state_representations = {
            s_id: self.active_window_state_representations[s_id]
            if s_id in self.active_window_state_representations
            else self.representation_constructor()
            for s_id in representation_ids
        }


class BaseBufferedAdaptiveLearner(BaseAdaptiveLearner):
    """A base adaptive learning class, using a buffer to stabilize observations."""

    def __init__(
        self,
        classifier_constructor: Callable[[], Classifier],
        representation_constructor: Callable[[int], ConceptRepresentation],
        representation_comparer: RepresentationComparer,
        drift_detector_constructor: Callable[[], DriftDetector],
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

        drift_detector_constructor: Callable[[], DriftDetector]
            A function to generate an object capable of detecting drift in a univariate stream.

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
            drift_detector_constructor,
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
        with pure_inference_mode():
            state_predictions = self.repository.get_repository_predictions(
                x, self.active_state_id, self.prediction_mode
            )
        p = self.combine_predictions(state_predictions)

        # Train unsupervised representation features
        # In the buffered version, we train on observations coming out
        # of the buffer, rather than new observations.
        self.buffer.buffer_unsupervised(x)
        stable_data = self.buffer.collect_stable_unsupervised()
        for stable_observation in stable_data:
            state_predictions = self.repository.get_repository_predictions(
                stable_observation.x, self.active_state_id, self.prediction_mode
            )
            self.train_components_unsupervised(x, state_predictions)

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

        # Train recent concept representation for active state
        self.train_components_supervised(x, y, sample_weight)

        self.step()

    def transition_reset(
        self, prev_active_state_id: int, next_active_state_id: int, in_drift: bool, in_warning: bool
    ) -> None:
        super().transition_reset(prev_active_state_id, next_active_state_id, in_drift, in_warning)
        self.buffer.reset_on_drift()
