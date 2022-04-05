""" Functions for performing classifier adaptation. The goal is to return
a state based on the current repository, background state and calculated state relevance.
Should return the background state to automatically construct a new state, or construct one here.
Note that new states should not be added to the repository or transitioned in this function. """

from typing import Dict, Optional

from streamselect.adaptive_learning.reidentification_schedulers import DriftInfo
from streamselect.repository import Repository
from streamselect.states import State


def maximum_relevance_adaptation(
    bg_state: Optional[State], repository: Repository, state_relevance: Dict[int, float], drift: Optional[DriftInfo]
) -> State:
    """Return the state from B u R with maximum relevance."""
    max_id, _ = max(state_relevance.items(), key=lambda x: x[1])
    if max_id in repository.states:
        adapted_state = repository.states[max_id]
    else:
        assert bg_state is not None
        adapted_state = bg_state
    return adapted_state
