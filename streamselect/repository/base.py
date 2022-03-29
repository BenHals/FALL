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
