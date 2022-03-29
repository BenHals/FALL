""" Base class for maintaining a repository of states. """

from typing import Dict

from streamselect.states import State

__all__ = ["Repository"]


class Repository:  # pylint: disable=too-few-public-methods
    """A base repository of states.
    Handles memory management.
    """

    def __init__(self) -> None:
        self.states: Dict[int, State] = {}

    def add(self, new_state: State) -> None:
        """Add a new state to the repository.
        Throws an error if the state already exists."""
        if new_state.state_id in self.states:
            raise ValueError(f"State with id {new_state.state_id} already exists.")

        self.states[new_state.state_id] = new_state

    def remove(self, state: State) -> None:
        """remove a state from the repository.
        Throws an error if the state does not exists."""
        if state.state_id not in self.states:
            raise ValueError(f"State with id {state.state_id} does not exist.")

        del self.states[state.state_id]
