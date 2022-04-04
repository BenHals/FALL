""" Base adaptive learning class. """
import abc
from collections import deque
from typing import Callable, Deque, Dict, Optional, Set, Tuple, Union

from river.base import Classifier, DriftDetector
from river.base.typing import ClfTarget
from river.utils import pure_inference_mode

from streamselect.adaptive_learning.buffer import SupervisedUnsupervisedBuffer
from streamselect.concept_representations import ConceptRepresentation
from streamselect.repository import Repository, RepresentationComparer, ValuationPolicy
from streamselect.states import State
from streamselect.utils import Observation, get_drift_detector_estimate


class PerformanceMonitor:
    def __init__(self) -> None:
        self.initial_active_state_id: int = -1
        self.in_drift: bool = False
        self.in_warning: bool = False
        self.made_transition: bool = False
        self.final_active_state_id: int = -1
        self.active_state_relevance: float = -1
        self.background_in_drift: bool = False
        self.background_in_warning: bool = False
        self.background_state_relevance: float = -1
        self.last_observation: Optional[Observation] = None

    def step_reset(self, initial_active_state_id: int) -> None:
        """Reset monitoring on taking a new step."""
        self.initial_active_state_id = initial_active_state_id
        self.in_drift = False
        self.in_warning = False
        self.made_transition = False
        self.final_active_state_id = -1
        self.active_state_relevance = -1
        self.background_in_drift = False
        self.background_in_warning = False
        self.background_state_relevance = -1
        self.last_observation = None

    def buffer_step_reset(self, initial_active_state_id: int) -> None:
        """Reset monitoring on taking a new step in a buffered classifier
        when no new stable data is available."""
        self.in_drift = False
        self.in_warning = False
        self.made_transition = False
        self.background_in_drift = False
        self.background_in_warning = False


class BaseAdaptiveLearner(Classifier, abc.ABC):
    """A base adaptive learning class."""

    def __init__(
        self,
        classifier_constructor: Callable[[], Classifier],
        representation_constructor: Callable[[int, int, str, int], ConceptRepresentation],
        representation_comparer: RepresentationComparer,
        drift_detector_constructor: Callable[[], DriftDetector],
        representation_update_period: int = 1,
        max_size: int = -1,
        valuation_policy: ValuationPolicy = ValuationPolicy.NoPolicy,
        train_representation: bool = True,
        window_size: int = 1,
        construct_pair_representations: bool = False,
        prediction_mode: str = "active",
        background_state_mode: Union[str, int, None] = "drift_reset",
        drift_detection_mode: str = "any",
    ) -> None:
        """
        Parameters
        ----------
        classifier_constructor: Callable[[], Classifier]
            A function to generate a new classifier.

        representation_constructor: Callable[[int, int, str, int], ConceptRepresentation]
            A function to generate a new concept representation taking in:
             window_size, state_id, mode and update_period.

        representation_comparer: RepresentationComparer
            An object capable of calculating similarity between two representations.

        drift_detector_constructor: Callable[[], DriftDetector]
            A function to generate an object capable of detecting drift in a univariate stream.

        representation_update_period: int
            Default: 1
            The number of timesteps between representation updates.

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

        drift_detection_mode: str["any", "lower", "higher]
            Default: "any"
            How change is interpreted as concept drift.
            "any": Any significant change in relevance is detected as drift.
            "lower": Significant changes where the new value is lower than the mean is detected.
            "higher": Significant changes where the new value is higher than the mean is detected.

        """
        self.representation_update_period = representation_update_period
        self.max_size = max_size
        self.valuation_policy = valuation_policy
        self.classifier_constructor = classifier_constructor
        self.active_representation_constructor = lambda state_id: representation_constructor(
            window_size, state_id, "active", representation_update_period
        )
        self.concept_representation_constructor = lambda state_id: representation_constructor(
            window_size, state_id, "concept", representation_update_period
        )
        self.representation_comparer = representation_comparer
        self.drift_detector_constructor = drift_detector_constructor
        self.train_representation = train_representation
        self.window_size = window_size
        self.construct_pair_representations = construct_pair_representations
        self.prediction_mode = prediction_mode
        self.background_state_mode = background_state_mode
        self.drift_detection_mode = drift_detection_mode

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
            representation_constructor=self.concept_representation_constructor,
            train_representation=self.train_representation,
        )
        self.drift_detector = self.drift_detector_constructor()
        active_state: State = self.repository.add_next_state()
        self.active_state_id: int = active_state.state_id

        self.active_window_state_representations: Dict[int, ConceptRepresentation] = {
            self.active_state_id: self.active_representation_constructor(self.active_state_id)
        }

        self.setup_background_state()

        self.performance_monitor = PerformanceMonitor()

    def setup_background_state(self) -> None:
        """Setup or reset the background state."""
        self.background_state: Optional[State] = None
        self.background_state_active_representation: Optional[ConceptRepresentation] = None
        self.background_state_detector: Optional[DriftDetector] = None
        if self.background_state_mode:
            self.background_state = self.repository.make_state(-1)
            self.background_state_active_representation = self.active_representation_constructor(
                self.background_state.state_id
            )
            if self.background_state_mode == "drift_reset":
                self.background_state_detector = self.drift_detector_constructor()

    def predict_one(self, x: dict, timestep: Optional[int] = None) -> ClfTarget:
        """Make a prediction using the active state classifier.
        Also trains unsupervised components of the classifier and concept representation.
        """
        unsupervised_observation = Observation(
            x,
            y=None,
            active_state_id=self.active_state_id,
            sample_weight=1.0,
            seen_at=timestep if timestep is not None else self.unsupervised_timestep,
        )

        state_predictions = self.repository.get_repository_predictions(unsupervised_observation, self.prediction_mode)
        p = self.combine_predictions(state_predictions)

        for state_id, state_p in state_predictions.items():
            unsupervised_observation.add_prediction(state_p, state_id)

        self.train_components_unsupervised(unsupervised_observation)
        self.train_background_unsupervised(unsupervised_observation)
        return p

    def combine_predictions(self, state_predictions: Dict[int, ClfTarget]) -> ClfTarget:
        """Combines state predictions into a single output prediction."""
        return state_predictions[self.active_state_id]

    def train_background_unsupervised(self, unsupervised_observation: Observation) -> None:
        """Train background state on unsupervised data.
        Skips training if background_mode is None."""
        if self.background_state:
            if self.background_state.state_id in self.repository.states:
                raise ValueError("A state with the same state_id has been added to the repository.")
            _ = self.background_state.predict_one(unsupervised_observation, force_train_own_representation=True)
            if self.background_state_active_representation:
                self.background_state_active_representation.predict_one(unsupervised_observation)

    def train_components_unsupervised(self, unsupervised_observation: Observation) -> None:
        """Train non-state components with unsupervised data."""
        # Train representations of recent data
        for _, representation in self.active_window_state_representations.items():
            representation.predict_one(unsupervised_observation)

        self.representation_comparer.train_unsupervised(self.repository)
        self.unsupervised_active_window.append(unsupervised_observation)

        self.unsupervised_timestep += 1

    def learn_one(self, x: dict, y: ClfTarget, sample_weight: float = 1.0, timestep: Optional[int] = None) -> None:
        active_state = self.get_active_state()

        supervised_observation = Observation(
            x,
            y=y,
            active_state_id=self.active_state_id,
            sample_weight=sample_weight,
            seen_at=timestep if timestep is not None else self.supervised_timestep,
        )

        # Train recent concept representation for active state
        self.train_components_supervised(supervised_observation)
        self.train_background_supervised(supervised_observation)

        # train supervised representation features and state classifier.
        trainable_states = self.repository.states.values() if self.construct_pair_representations else [active_state]
        for state in trainable_states:
            state.learn_one(supervised_observation)

        self.step(supervised_observation)

    def train_background_supervised(self, supervised_observation: Observation) -> None:
        """Train background state on supervised data.
        Skips training if background_mode is None."""
        if self.background_state:
            with pure_inference_mode():
                # The background state should treat itself as active, i.e., train its own representation and classifier
                # thus the forced training.
                _ = self.background_state.predict_one(supervised_observation, force_train_own_representation=True)
                self.background_state.learn_one(supervised_observation, force_train_classifier=True)
            if self.background_state_active_representation:
                self.background_state_active_representation.learn_one(supervised_observation)
            self.background_state.step(supervised_observation.sample_weight, is_active=True)

    def train_components_supervised(self, supervised_observation: Observation) -> None:
        """Train non-state components with supervised data."""
        with pure_inference_mode():
            _ = self.repository.get_repository_predictions(supervised_observation, self.prediction_mode)

        for _, representation in self.active_window_state_representations.items():
            representation.learn_one(supervised_observation)

        self.representation_comparer.train_supervised(self.repository)
        self.supervised_active_window.append(supervised_observation)
        self.supervised_timestep += 1

    def step(self, observation: Observation) -> None:
        """Update internal state"""
        self.performance_monitor.step_reset(self.active_state_id)

        self.performance_monitor.last_observation = observation

        # Update state statistics
        self.repository.step_all(self.active_state_id)

        in_drift, in_warning, active_state_relevance = self.active_state_drift_detection()
        self.performance_monitor.in_drift = in_drift
        self.performance_monitor.in_warning = in_warning
        self.performance_monitor.active_state_relevance = active_state_relevance

        if in_drift:
            state_relevance = self.perform_reidentification()
            new_active_state = self.get_adapted_state(state_relevance)
            if new_active_state != self.get_active_state():
                self.transition_active_state(new_active_state, True, in_warning)
                self.performance_monitor.made_transition = True

        self.evaluate_background_state(transitioned=self.performance_monitor.made_transition)
        self.performance_monitor.final_active_state_id = self.active_state_id

    def evaluate_background_state(self, transitioned: bool) -> None:
        """Step the background state, and reset if needed."""
        if not self.background_state:
            return

        reset_required = False
        if self.background_state_mode == "drift_reset":
            if self.background_state_active_representation and self.background_state_detector:
                b_in_drift, b_in_warning, b_relevance = self.perform_drift_detection(
                    self.background_state, self.background_state_active_representation, self.background_state_detector
                )
                self.performance_monitor.background_in_drift = b_in_drift
                self.performance_monitor.background_in_warning = b_in_warning
                self.performance_monitor.background_state_relevance = b_relevance
                reset_required = b_in_drift
        elif isinstance(self.background_state_mode, int):
            reset_required = self.background_state.seen_weight > self.background_state_mode

        if transitioned:
            reset_required = True

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
        return self.perform_drift_detection(active_state, active_representation, self.drift_detector)

    def perform_drift_detection(
        self,
        state: State,
        state_representation: ConceptRepresentation,
        drift_detector: DriftDetector,
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

        in_drift, in_warning = drift_detector.update(state_relevance)  # type: ignore

        # turn off detections which do not match mode
        if self.drift_detection_mode == "lower":
            if state_relevance >= get_drift_detector_estimate(drift_detector):
                in_drift = False
                in_warning = False
        if self.drift_detection_mode == "higher":
            if state_relevance <= get_drift_detector_estimate(drift_detector):
                in_drift = False
                in_warning = False

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
        representation = self.active_representation_constructor(state.state_id)
        supervised_timesteps = set()
        if mode in ["supervised", "both"]:
            for observation in self.get_supervised_active_window():
                if observation.y is None:
                    continue
                supervised_timesteps.add(observation.seen_at)
                with pure_inference_mode():
                    _ = state.predict_one(observation)
                    representation.predict_one(observation)
                representation.learn_one(observation)

        if mode in ["unsupervised", "both"]:
            for observation in self.get_unsupervised_active_window():
                if observation.seen_at in supervised_timesteps:
                    continue
                with pure_inference_mode():
                    _ = state.predict_one(observation)
                representation.predict_one(observation)

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
        of previous states. Adds the state to the repository."""
        max_state_id, _ = max(state_relevance.items(), key=lambda x: x[1])
        if max_state_id in [-1, self.active_state_id]:
            # We skip memory management so that we may use states while processing
            # the transition. We manually call apply_memory_management at the end of
            # the transition to handle this after the transition.
            new_state = self.repository.add_next_state(skip_memory_management=True)
        else:
            new_state = self.repository.states[max_state_id]
        return new_state

    def transition_active_state(self, next_active_state: State, in_drift: bool, in_warning: bool) -> None:
        """Transition to an active state in the repository."""
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
            else self.active_representation_constructor(s_id)
            for s_id in representation_ids
        }
        self.drift_detector.reset()
        self.repository.apply_memory_management()


def get_constant_max_buffer_scheduler() -> Callable[[float, State, Optional[Observation]], float]:
    """Returns a buffer timeout scheduler which always
    sets the buffer_timeout to be max."""

    def get_buffer_timeout(
        buffer_timeout_max: float, active_state: State, observation: Optional[Observation] = None
    ) -> float:
        return buffer_timeout_max

    return get_buffer_timeout


def get_increasing_buffer_scheduler(
    increase_rate: float = 1.0,
) -> Callable[[float, State, Optional[Observation]], float]:
    """Returns a buffer timeout scheduler which always
    sets the buffer_timeout to be max."""

    def get_buffer_timeout(
        buffer_timeout_max: float, active_state: State, observation: Optional[Observation] = None
    ) -> float:
        return min(round(active_state.seen_weight * increase_rate), buffer_timeout_max)

    return get_buffer_timeout


class BaseBufferedAdaptiveLearner(BaseAdaptiveLearner):
    """A base adaptive learning class, using a buffer to stabilize observations."""

    def __init__(
        self,
        classifier_constructor: Callable[[], Classifier],
        representation_constructor: Callable[[int, int, str, int], ConceptRepresentation],
        representation_comparer: RepresentationComparer,
        drift_detector_constructor: Callable[[], DriftDetector],
        representation_update_period: int = 1,
        max_size: int = -1,
        valuation_policy: ValuationPolicy = ValuationPolicy.NoPolicy,
        train_representation: bool = True,
        window_size: int = 1,
        buffer_timeout_max: float = 0.0,
        construct_pair_representations: bool = False,
        prediction_mode: str = "active",
        background_state_mode: Union[str, int, None] = "drift_reset",
        drift_detection_mode: str = "any",
        buffer_timeout_scheduler: Callable[
            [float, State, Optional[Observation]], float
        ] = get_increasing_buffer_scheduler(1.0),
    ) -> None:
        """
        Parameters
        ----------
        classifier_constructor: Callable[[], Classifier]
            A function to generate a new classifier.

        representation_constructor: Callable[[int, int, str, int], ConceptRepresentation]
            A function to generate a new concept representation taking in:
             window_size, state_id, mode and update_period.

        representation_comparer: RepresentationComparer
            An object capable of calculating similarity between two representations.

        drift_detector_constructor: Callable[[], DriftDetector]
            A function to generate an object capable of detecting drift in a univariate stream.

        representation_update_period: int
            Default: 1
            The number of timesteps between representation updates.

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

        buffer_timeout_max: float
            The max number of timesteps to buffer new data before training.

        construct_pair_representations: bool
            Whether or not to construct a representation for each concept, classifier pair.
            Such a pair R_{ij} represents data drawn from concept j classified by state i.
            If False, only constructs R_{ii}, which is fine for many adaptive learning systems.

        buffer_timeout_scheduler: Callable[[float, State, Observation], float
            Default: constant_max_buffer_scheduler
            A function to calculate the current buffer_timeout. The function is passed buffer_timeout_max,
            the active state and (optinally) the new observation.
            The default simply returns the buffer_timeout_max. An alternative is the increasing
            scheduler, which slowly increases the buffer_timeout so that a new classifier may learn.

        """
        super().__init__(
            classifier_constructor,
            representation_constructor,
            representation_comparer,
            drift_detector_constructor,
            representation_update_period,
            max_size,
            valuation_policy,
            train_representation,
            window_size,
            construct_pair_representations,
            prediction_mode,
            background_state_mode,
            drift_detection_mode,
        )
        self.buffer_timeout_max = buffer_timeout_max
        self.buffer_timeout_scheduler = buffer_timeout_scheduler
        self.buffer_timeout = self.buffer_timeout_scheduler(self.buffer_timeout_max, self.get_active_state(), None)

        # Create a buffer to hold supervised and unsupervised data.
        # Uses a clock based on supervised data, i.e., supervised data is considered stable when it is older
        # than buffer_timout supervised observations, and all unsupervised observations collected
        # before this observation is also released.
        self.buffer = SupervisedUnsupervisedBuffer(
            self.window_size, self.buffer_timeout, self.buffer_timeout, release_strategy="supervised"
        )

    def predict_one(self, x: dict, timestep: Optional[int] = None) -> ClfTarget:
        """Make a prediction using the active state classifier.
        Also trains unsupervised components of the classifier and concept representation.

        Note: A buffer is used to only train on stable"""

        unsupervised_observation = Observation(
            x,
            y=None,
            active_state_id=self.active_state_id,
            sample_weight=1.0,
            seen_at=timestep if timestep is not None else self.unsupervised_timestep,
        )

        with pure_inference_mode():
            state_predictions = self.repository.get_repository_predictions(
                unsupervised_observation, self.prediction_mode
            )
        p = self.combine_predictions(state_predictions)

        # Train unsupervised representation features
        # In the buffered version, we train on observations coming out
        # of the buffer, rather than new observations.
        self.buffer.buffer_unsupervised(x, unsupervised_observation.active_state_id, unsupervised_observation.seen_at)
        stable_data = self.buffer.collect_stable_unsupervised()
        for stable_observation in stable_data:
            _ = self.repository.get_repository_predictions(stable_observation, self.prediction_mode)

        self.train_components_unsupervised(unsupervised_observation)
        self.train_background_unsupervised(unsupervised_observation)

        return p

    def learn_one(self, x: dict, y: ClfTarget, sample_weight: float = 1.0, timestep: Optional[int] = None) -> None:
        self.performance_monitor.buffer_step_reset(self.active_state_id)

        active_state = self.get_active_state()

        supervised_observation = Observation(
            x,
            y=y,
            active_state_id=self.active_state_id,
            sample_weight=sample_weight,
            seen_at=timestep if timestep is not None else self.supervised_timestep,
        )
        # Train recent concept representation for active state
        self.train_components_supervised(supervised_observation)
        self.train_background_supervised(supervised_observation)

        self.set_buffer_timeout(
            self.buffer_timeout_scheduler(self.buffer_timeout_max, self.get_active_state(), supervised_observation)
        )
        self.buffer.buffer_supervised(
            x,
            y,
            supervised_observation.active_state_id,
            supervised_observation.seen_at,
            supervised_observation.sample_weight,
        )
        stable_data = self.buffer.collect_stable_supervised()

        # train supervised representation features and state classifier.
        # In the buffered version, we train on observations coming out
        # of the buffer, rather than new observations.
        for stable_observation in stable_data:
            if stable_observation.y is None:
                continue

            trained_states = self.repository.states.values() if self.construct_pair_representations else [active_state]
            for state in trained_states:
                state.learn_one(stable_observation)

            self.step(stable_observation)

    def transition_reset(
        self, prev_active_state_id: int, next_active_state_id: int, in_drift: bool, in_warning: bool
    ) -> None:
        super().transition_reset(prev_active_state_id, next_active_state_id, in_drift, in_warning)
        self.buffer.reset_on_drift(next_active_state_id)

    def set_buffer_timeout(self, buffer_timeout: float) -> None:
        """Set a new buffer timeout."""
        self.buffer_timeout = buffer_timeout
        self.buffer.supervised_buffer_timeout = buffer_timeout
        self.buffer.unsupervised_buffer_timeout = buffer_timeout
