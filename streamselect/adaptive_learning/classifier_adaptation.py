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
) -> Optional[State]:
    """Return the state from B u R with maximum relevance."""
    max_id, _ = max(state_relevance.items(), key=lambda x: x[1])
    if max_id in repository.states:
        adapted_state = repository.states[max_id]
    else:
        assert bg_state is not None
        adapted_state = bg_state
    return adapted_state


def max_acc_sig_relevance_adaptation(
    bg_state: Optional[State], repository: Repository, state_relevance: Dict[int, float], drift: Optional[DriftInfo]
) -> Optional[State]:
    """Get the set of states from B u R with maximum relevance within some significance threshold,
    and return the most accurate."""
    # accept states within threshold_stdev standard deviation of the max relevance.
    threshold_stdev = 0.5

    max_id, max_relevance = max(state_relevance.items(), key=lambda x: x[1])
    if max_id == -1:
        assert bg_state is not None
        max_stdev = bg_state.in_concept_relevance_distribution.stdev
    else:
        max_stdev = repository.states[max_id].in_concept_relevance_distribution.stdev

    accepted_states = []
    for state_id, relevance in state_relevance.items():
        if state_id == -1:
            assert bg_state is not None
            state = bg_state
        else:
            state = repository.states[state_id]
        state_stdev = state.in_concept_relevance_distribution.stdev
        state_mean = state.in_concept_relevance_distribution.mean

        if (
            relevance >= max_relevance - (max_stdev * threshold_stdev)
            or max_relevance <= relevance + (state_stdev * threshold_stdev)
        ) and relevance > state_mean - (state_stdev * threshold_stdev):
            accepted_states.append(state)

    if len(accepted_states) == 0:
        adapted_state = None

    elif len(accepted_states) == 1:
        adapted_state = accepted_states[0]
    else:
        adapted_state = max(accepted_states, key=lambda x: x.in_concept_accuracy_record.estimation)

    return adapted_state
