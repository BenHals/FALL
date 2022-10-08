""" Base adaptive learning class. """
import abc
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple, Union

from river.base import Classifier, DriftDetector
from river.base.typing import ClfTarget
from river.compose import pure_inference_mode

from streamselect.adaptive_learning.buffer import SupervisedUnsupervisedBuffer
from streamselect.adaptive_learning.classifier_adaptation import (
    max_acc_sig_relevance_adaptation,
    maximum_relevance_adaptation,
)
from streamselect.adaptive_learning.reidentification_schedulers import (
    BaseReidentificationScheduler,
    DriftInfo,
    DriftType,
    ReidentificationSchedule,
)
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
        self.active_state_last_relevance: float = -1
        self.background_in_drift: bool = False
        self.background_in_warning: bool = False
        self.background_state_relevance: float = -1
        self.last_observation: Optional[Observation] = None
        self.last_trained_observation: Optional[Observation] = None
        self.last_drift: Optional[DriftInfo] = None
        self.deletions: list[int] = []
        self.merges: dict[int, int] = {}
        self.repository: dict[int, State] = {}
        self.concept_occurences: dict[str, int] = {}
        self.transition_matrix: dict[str, dict[str, int]] = {}
        self.state_relevances: dict[int, float] = {}

    def step_reset(self, initial_active_state: State) -> None:
        """Reset monitoring on taking a new step."""
        self.set_initial_active_state(initial_active_state)
        self.in_drift = False
        self.in_warning = False
        self.made_transition = False
        self.final_active_state_id = -1
        self.active_state_last_relevance = -1
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

    def set_initial_active_state(self, initial_active_state: State) -> None:
        self.initial_active_state_id = initial_active_state.state_id
        if self.initial_active_state_id not in self.repository:
            self.repository[self.initial_active_state_id] = initial_active_state

    def set_final_active_state(self, final_active_state: State) -> None:
        self.final_active_state_id = final_active_state.state_id
        if self.final_active_state_id not in self.repository:
            self.repository[self.final_active_state_id] = final_active_state

    
    def add_to_transition_matrix(self, init_s: int, curr_s: int, matrix: dict[str, dict[str, int]], weight: int=1) -> dict[str, dict[str, int]]:
        matrix[str(init_s)][str(curr_s)] = matrix[str(init_s)].get(str(curr_s), 0) + weight
        matrix[str(init_s)]['total'] += weight
        return matrix

    def record_transition(self, initial_state: int, final_state: int) -> None:
        """ Record an observation to observation level transition from the initial_state to the current state.
        Depends on if a drift was detected, or if in warning which records are updated.
        """
        if initial_state in self.deletions or initial_state is None:
            if initial_state in self.merges:
                while initial_state in self.merges:
                    initial_state = self.merges[initial_state]
            else:
                raise ValueError("Recording transition from deleted state")

        created_new_state = False
        if str(initial_state) not in self.transition_matrix:
            self.transition_matrix[str(initial_state)] = {}
            self.transition_matrix[str(initial_state)]['total'] = 0
        if str(final_state) not in self.transition_matrix:
            self.transition_matrix[str(final_state)] = {}
            self.transition_matrix[str(final_state)]['total'] = 0
            created_new_state = True

        self.add_to_transition_matrix(initial_state, final_state, self.transition_matrix)

    def delete_merge_state(self, merge_from: str, merge_into: str) -> None:
        # Handle merging transition states, which are
        # not in the repository
        if 'T' not in str(merge_from):
            self.repository.pop(int(merge_from), 0)
            self.deletions.append(int(merge_from))

        # Need to merge from transition matrix
        # First merge transitions from merge_from into those from
        transitions_from = self.transition_matrix.pop(str(merge_from), {})
        new_transitions = self.transition_matrix.setdefault(str(merge_into), {'total': 0})
        for to_id in transitions_from:
            if to_id == "total":
                continue
            n_trans_merge = transitions_from[to_id]
            n_trans_into = new_transitions.setdefault(to_id, 0)
            new_transitions[to_id] = n_trans_merge + n_trans_into
            new_transitions['total'] += n_trans_merge

        # Then merge transitions into this state
        # We delete the entry and add to entry for merge_into
        for from_state in list(self.transition_matrix.keys()):
            n_trans_merge = self.transition_matrix[from_state].pop(str(merge_from), 0)
            n_trans_to = self.transition_matrix[from_state].setdefault(str(merge_into), 0)
            self.transition_matrix[from_state][str(merge_into)] = n_trans_merge + n_trans_to

        merge_from_occurences = self.concept_occurences.pop(merge_from, 0)
        self.concept_occurences[merge_into] = self.concept_occurences.get(merge_into, 0) + merge_from_occurences

    def get_prev_state_from_transitions(self, state_id: int) -> tuple[int, str, str]:
        possible_previous_states: list[tuple[int, str]] = [(0, 'T')]
        for prev_id in self.repository.keys():
            if prev_id == state_id or prev_id not in self.transition_matrix:
                continue
            if state_id in self.transition_matrix[str(prev_id)]:
                n_trans = self.transition_matrix[str(prev_id)][str(state_id)]
                possible_previous_states.append((n_trans, str(prev_id)))
        prev_state_count, prev_state = max(possible_previous_states, key=lambda x: x[0])
        prev_state_transition = f"T-{prev_state}"
        return (prev_state_count, prev_state, prev_state_transition)

    def delete_into_transition_state(self, delete_id: int) -> str:
        """ Merge a state into a generic transition state following the previous state.
        In many cases, i.e., gradual drift between state 0 -> state 1, there will be some
        transition state between them which has no meaning. We don't want to store all of these
        states, as they have no individual meaning, but we want to store the idea that there is
        some generic transition following state 0 and going to state 1. S0 -> T0 -> S1.

        So if we see S0->S2->S1 and decide S2 was only a transition, we delete it and merge its
        transitions into T-0.

        Returns what the delete_id turned into so we can update transitions.
        (e.g., returns T-0 rather than 2)
        """

        # First we need to work out what the previous state was, as we don't store this
        # (maybe we should).
        # We can find this as an entry in the transition matrix.
        # Should only be one, but we handle if there are multiple. We take the max num transitions
        # to be the previous state
        prev_state_count, prev_state, prev_state_transition = self.get_prev_state_from_transitions(delete_id)

        init_state = prev_state_transition
        self.delete_merge_state(str(delete_id), init_state)

        # We create a transition state for all states, so need to also
        # merge the one for the delete state.
        self.delete_merge_state(f"T-{delete_id}", init_state)

        return init_state

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
        representation_window_size: int = 25,
        construct_pair_representations: bool = False,
        prediction_mode: str = "active",
        background_state_mode: Union[str, int, None] = "drift_reset",
        drift_detection_mode: str = "lower",
        reidentification_check_schedulers: Optional[List[BaseReidentificationScheduler]] = None,
    ) -> None:
        """
        Parameters
        ----------
        classifier_constructor: Callable[[], Classifier]
            A function to generate a new classifier.

        representation_constructor: Callable[[int, int, str, int], ConceptRepresentation]
            A function to generate a new concept representation taking in:
             representation_window_size, state_id, mode and update_period.

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

        representation_window_size: int
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

        reidentification_check_schedulers: Optional[List[BaseReidentificationScheduler]]
            Default: None
            A list of schedulers to perform automated active state checks.
            For example, a DriftDetectionCheck(check_delay) can be passed to perform
            a re-identification check check_delay timesteps after each detection in order
            to confirm the optimal state was selected.

        """
        self.representation_update_period = representation_update_period
        self.max_size = max_size
        self.valuation_policy = valuation_policy
        self.classifier_constructor = classifier_constructor
        self.active_representation_constructor = lambda state_id: representation_constructor(
            representation_window_size, state_id, "active", representation_update_period
        )
        self.concept_representation_constructor = lambda state_id: representation_constructor(
            representation_window_size, state_id, "concept", representation_update_period
        )
        self.representation_comparer = representation_comparer
        self.drift_detector_constructor = drift_detector_constructor
        self.train_representation = train_representation
        self.representation_window_size = representation_window_size
        self.construct_pair_representations = construct_pair_representations
        self.prediction_mode = prediction_mode
        self.background_state_mode = background_state_mode
        self.drift_detection_mode = drift_detection_mode
        self.reidentification_check_schedulers = (
            reidentification_check_schedulers if reidentification_check_schedulers is not None else []
        )
        self.classifier_adaptation_mode = "MASR"

        classifier_adaptation_mode_map = {"MASR": max_acc_sig_relevance_adaptation, "MR": maximum_relevance_adaptation}

        # self.perform_classifier_adaptation = maximum_relevance_adaptation
        try:
            self.perform_classifier_adaptation = classifier_adaptation_mode_map[
                self.classifier_adaptation_mode.upper()
            ]
        except KeyError as e:
            raise ValueError(f"Classifier adaptation mode {self.classifier_adaptation_mode} not recognized. ") from e

        # Validation
        if self.prediction_mode != "all" and self.construct_pair_representations:
            raise ValueError(
                "Prediction mode not set to all, but construct_pair_representation requires all predictions."
            )

        # timestep for unsupervised data
        self.supervised_timestep = 0
        # timestep for supervised data
        self.unsupervised_timestep = 0

        self.unsupervised_active_window: Deque[Observation] = deque(maxlen=self.representation_window_size)
        self.supervised_active_window: Deque[Observation] = deque(maxlen=self.representation_window_size)

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

        self.reidentification_schedule = ReidentificationSchedule()
        for scheduler in self.reidentification_check_schedulers:
            self.reidentification_schedule.add_scheduler(scheduler)
        self.reidentification_schedule.initialize(0)

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

        unsupervised_observation.is_stable = True
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
        supervised_observation.is_stable = True
        trainable_states = self.repository.states.values() if self.construct_pair_representations else [active_state]
        for state in trainable_states:
            state.learn_one(supervised_observation)
        self.performance_monitor.last_trained_observation = supervised_observation

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
        active_state = self.get_active_state()
        self.performance_monitor.step_reset(active_state)

        self.performance_monitor.last_observation = observation

        # Update state statistics
        self.repository.step_all(self.active_state_id)

        in_drift, in_warning, active_state_relevance = self.active_state_drift_detection()
        observation.add_active_state_relevance(active_state_relevance, self.active_state_id)
        # if the observation is stable then we already trained on it, so we should add
        # relevance now.
        if observation.is_stable:
            active_state.add_active_state_relevance(active_state_relevance)
        self.performance_monitor.in_drift = in_drift
        self.performance_monitor.in_warning = in_warning
        self.performance_monitor.active_state_last_relevance = active_state_relevance
        self.performance_monitor.state_relevances[active_state.state_id] = active_state.get_in_concept_relevance()

        # Check if we need to perform reidentification
        # either from a scheduled check for from a drift detection.
        step_reidentification_checks: List[DriftInfo] = []
        scheduled_checks = self.reidentification_schedule.get_scheduled_reidentifications(int(observation.seen_at))
        if scheduled_checks is not None:
            step_reidentification_checks += scheduled_checks
        if in_drift:
            step_reidentification_checks.append(
                DriftInfo(int(observation.seen_at), drift_type=DriftType.DriftDetectorTriggered)
            )

        # Schedule any new re-identification checks.
        for drift in step_reidentification_checks:
            self.reidentification_schedule.schedule_reidentification(drift)

        if len(step_reidentification_checks) > 0:
            # We always use the drift detector drift if availiable, or the newest scheduled drift if not.
            # This is based on the order, where scheduled drifts are expected to be ordered by time
            # then any detection is added last.
            drift = step_reidentification_checks[-1]
            self.performance_monitor.last_drift = drift
            state_relevance = self.perform_reidentification(drift)
            drift.reidentification_relevance = state_relevance
            new_active_state = self.get_adapted_state(state_relevance, drift)
            if new_active_state != self.get_active_state():
                drift.triggered_transition = True
                drift.transitioned_from = self.active_state_id
                drift.transitioned_to = new_active_state.state_id
                self.transition_active_state(new_active_state, True, in_warning)
                self.performance_monitor.made_transition = True

        self.evaluate_background_state(transitioned=self.performance_monitor.made_transition)
        self.performance_monitor.set_final_active_state(self.get_active_state())

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
        in_drift, in_warning, active_state_relevance = self.perform_drift_detection(active_state, active_representation, self.drift_detector)
        active_state.add_active_state_relevance(active_state_relevance)
        return in_drift, in_warning, active_state_relevance

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

        _ = drift_detector.update(state_relevance)  # type: ignore
        in_drift = drift_detector.drift_detected
        in_warning = False
        # turn off detections which do not match mode
        if in_drift:
            print(state_relevance, get_drift_detector_estimate(drift_detector))
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

    def perform_reidentification(self, drift: DriftInfo) -> Dict[int, float]:
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

    def get_adapted_state(self, state_relevance: Dict[int, float], drift: DriftInfo) -> State:
        """Returns a new state adapted to current conditions, based on estimated relevance
        of previous states. Adds the state to the repository."""

        # Get the adapted state based on state_relevance.
        # May be a state from the repository, the background state, the active state
        # or a newly created state.
        adapted_state = self.perform_classifier_adaptation(
            self.background_state, self.repository, state_relevance, drift
        )

        # Setup IDs which trigger a new state to be created.
        # If we are just checking reidentification, this is only the background state.
        # If we are reacting to a drift detection trigger, we also transition to a new state
        # if the current active state is most relevant, as the drift detection indicates it
        # is no longer relevant.
        new_state_triggers: List[int] = [-1]
        if drift.drift_type == DriftType.DriftDetectorTriggered:
            new_state_triggers.append(self.active_state_id)

        # if the selected state was in the new_state_triggers, we instead select
        # a newly constructed state.
        if adapted_state.state_id in new_state_triggers:
            adapted_state = self.repository.make_next_state()

        if adapted_state.state_id not in self.repository.states:
            # We skip memory management for now so that we may use states while processing
            # the transition. We manually call apply_memory_management at the end of
            # the transition to handle this after the transition.
            self.repository.add(adapted_state, skip_memory_management=True)

        return adapted_state

    def transition_active_state(self, next_active_state: State, in_drift: bool, in_warning: bool) -> None:
        """Transition to an active state in the repository."""
        current_active_state = self.get_active_state()
        self.active_state_id = next_active_state.state_id
        self.repository.add_transition(
            current_active_state, next_active_state, in_drift=in_drift, in_warning=in_warning
        )
        self.performance_monitor.record_transition(current_active_state.state_id, next_active_state.state_id)

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
        self.drift_detector._reset()
        self.repository.apply_memory_management()
        self.reidentification_schedule.transition_reset(self.supervised_timestep)


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
    """Returns a buffer timeout scheduler which increases the buffer time as more weight is seen."""

    def get_buffer_timeout(
        buffer_timeout_max: float, active_state: State, observation: Optional[Observation] = None
    ) -> float:
        return min(round(active_state.active_seen_weight * increase_rate), buffer_timeout_max)

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
        representation_window_size: int = 25,
        buffer_timeout_max: float = 0.0,
        construct_pair_representations: bool = False,
        prediction_mode: str = "active",
        background_state_mode: Union[str, int, None] = "drift_reset",
        drift_detection_mode: str = "lower",
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
             representation_window_size, state_id, mode and update_period.

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

        representation_window_size: int
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
            representation_window_size,
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
            self.representation_window_size, self.buffer_timeout, self.buffer_timeout, release_strategy="supervised"
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
        unsupervised_observation = self.buffer.buffer_unsupervised(
            x, unsupervised_observation.active_state_id, unsupervised_observation.seen_at
        )
        stable_data = self.buffer.collect_stable_unsupervised()
        for stable_observation in stable_data:
            stable_observation.is_stable = True
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
        self.set_buffer_timeout(
            self.buffer_timeout_scheduler(self.buffer_timeout_max, self.get_active_state(), supervised_observation)
        )
        supervised_observation = self.buffer.buffer_supervised(
            x,
            y,
            supervised_observation.active_state_id,
            supervised_observation.seen_at,
            supervised_observation.sample_weight,
        )
        # Train recent concept representation for active state
        self.train_components_supervised(supervised_observation)
        self.train_background_supervised(supervised_observation)

        stable_data = self.buffer.collect_stable_supervised()

        # train supervised representation features and state classifier.
        # In the buffered version, we train on observations coming out
        # of the buffer, rather than new observations.
        for stable_observation in stable_data:
            stable_observation.is_stable = True
            if stable_observation.y is None:
                continue

            trained_states = self.repository.states.values() if self.construct_pair_representations else [active_state]
            for state in trained_states:
                state.learn_one(stable_observation)
            self.performance_monitor.last_trained_observation = stable_observation

        self.step(supervised_observation)

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
