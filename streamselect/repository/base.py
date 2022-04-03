""" Base class for maintaining a repository of states. """

from enum import Enum
from typing import Callable, Dict, Union

from river.base import Classifier
from river.base.typing import ClfTarget

from streamselect.concept_representations import ConceptRepresentation
from streamselect.repository.transition_fsm import TransitionFSM
from streamselect.states import State
from streamselect.utils import Observation

__all__ = ["Repository"]


class ValuationPolicy(Enum):
    NoPolicy = (0,)
    FIFO = 1
    LRU = 2
    Accuracy = 3


class Repository:  # pylint: disable=too-few-public-methods
    """A base repository of states.
    Handles memory management.
    """

    def __init__(
        self,
        max_size: int = -1,
        valuation_policy: ValuationPolicy = ValuationPolicy.NoPolicy,
        classifier_constructor: Union[Callable[[], Classifier], None] = None,
        representation_constructor: Union[Callable[[int], ConceptRepresentation], None] = None,
        train_representation: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        max_size: int
            Default -1
            Maximum number of states able to be stored.
            -1 represents unlimited. Anything else requires
            a deletion valuation policy.

        valuation_policy: ValiationPolicy
        Default NoPolicy
            The valuation policy to use. Must be set if max_size != -1

        classifier_constructor: Union[Callable[[], Classifier], None]
            A function to generate a new classifier, or none.
            Must be set to a valid constructor to automatically construct states.

        representation_constructor: Union[Callable[[int], ConceptRepresentation], None]
            A function which takes a state_id to generate a new concept representation, or none.
            Must be set to a valid constructor to automatically construct states.

        train_representation: bool
            Whether or not new states train representations.
            Must be set to automatically construct states.

        """
        self.max_size = max_size
        self.valuation_policy = valuation_policy
        self.classifier_constructor = classifier_constructor
        self.representation_constructor = representation_constructor
        self.train_representation = train_representation

        self.next_id = 0

        self.states: Dict[int, State] = {}
        self.base_transitions = TransitionFSM()
        self.warning_transitions = TransitionFSM()
        self.drift_transitions = TransitionFSM()
        self.all_transitions = [self.base_transitions, self.warning_transitions, self.drift_transitions]

    def make_state(self, state_id: int) -> State:
        """Construct a new state."""
        if self.classifier_constructor is None or self.representation_constructor is None:
            raise ValueError("Cannot construct state without setting valid constructors")
        return State(
            self.classifier_constructor(), self.representation_constructor, state_id, self.train_representation
        )

    def add_next_state(self, skip_memory_management: bool = False) -> State:
        """Create and add a state with the next valid ID.
        Return this state. Can only use if classifier constructor is set.
        Parameters
        ----------

        skip_memory_management: bool
            Default: False
            Whether or not to skip memory management.
            Skipping may be useful to aid in transitioning away from the prev state,
            but apply_memory_management should be manually called after this is done."""

        state = self.make_state(self.next_id)
        self.add(state, skip_memory_management=skip_memory_management)
        self.next_id += 1

        return state

    def add(self, new_state: State, skip_memory_management: bool = False) -> None:
        """Add a new state to the repository.
        Throws an error if the state already exists.
        Parameters
        ----------
        new_state: State
            The new state to add.

        skip_memory_management: bool
            Default: False
            Whether or not to skip memory management.
            Skipping may be useful to aid in transitioning away from the prev state,
            but apply_memory_management should be manually called after this is done.
        """
        if new_state.state_id in self.states:
            raise ValueError(f"State with id {new_state.state_id} already exists.")

        self.states[new_state.state_id] = new_state

        if not skip_memory_management:
            self.apply_memory_management()

    def apply_memory_management(self) -> None:
        """Apply memory management to avoid storing more than self.max_size states."""
        while len(self.states) > self.max_size and self.max_size != -1:
            self.memory_management_deletion()

    def add_transition(
        self, from_state: State, to_state: State, weight: int = 1, in_drift: bool = False, in_warning: bool = False
    ) -> None:
        """Add a transition to the relevant finite state machines."""
        self.base_transitions.add_transition(from_state.state_id, to_state.state_id, weight)
        if in_drift:
            self.drift_transitions.add_transition(from_state.state_id, to_state.state_id, weight)
        elif in_warning:
            self.warning_transitions.add_transition(from_state.state_id, to_state.state_id, weight)

    def remove(self, state: State) -> None:
        """remove a state from the repository.
        Throws an error if the state does not exists."""
        if state.state_id not in self.states:
            raise ValueError(f"State with id {state.state_id} does not exist.")

        del self.states[state.state_id]
        for transitions in self.all_transitions:
            transitions.delete_state(state.state_id)

    def step_all(self, active_state_id: int, sample_weight: float = 1.0) -> None:
        """Call step on all states to update statistics."""
        for state_id, state in self.states.items():
            state.step(sample_weight, is_active=state_id == active_state_id)

    def get_repository_predictions(self, observation: Observation, prediction_mode: str) -> Dict[int, ClfTarget]:
        """Makes a prediction for states in the repository.
        Returns
        -------
            state_predictions: Dict[int, ClfTarget]
                A dict of state_ids to their predictions.
                Contains all states if prediction_mode == "all"
                    otherwise only the active state prediciton."""
        state_predictions: Dict[int, ClfTarget] = {}
        active_state = self.states[observation.active_state_id]
        states_to_predict = self.states.values() if prediction_mode == "all" else [active_state]
        for state in states_to_predict:
            p = state.predict_one(observation)
            state_predictions[state.state_id] = p
        return state_predictions

    def memory_management_deletion(self) -> None:
        """Process the deletion of a single state.
        We delete the state with minimum value with regards to the
        valuation_policy parameter."""
        del_state_id = min(self.states, key=self.apply_valuation_policy)
        self.remove(self.states[del_state_id])

    def apply_valuation_policy(self, state_id: int) -> float:
        """Apply the current valuation policy to a state_id.
        Returns the valuation.

        Note
        ----
        The valuation should be a float representing the estimated benefit
        provided by the state in the future."""
        state = self.states[state_id]
        strategy = None
        if self.valuation_policy == ValuationPolicy.FIFO:
            strategy = self.fifo_valuation
        elif self.valuation_policy == ValuationPolicy.LRU:
            strategy = self.lru_valuation

        if strategy is None:
            raise ValueError("Valuation required but no valid valuation policy specified.")

        return strategy(state)

    def fifo_valuation(self, state: State) -> float:  # pylint: disable=R0201
        """FIFO policies value states by their age, with a more recent
        age valued higher."""
        return -1 * state.seen_weight

    def lru_valuation(self, state: State) -> float:  # pylint: disable=R0201
        """LRU policies value states by the time since last use, with a more recent
        use valued higher."""
        return -1 * state.weight_since_last_active
