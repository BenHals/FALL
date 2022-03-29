from streamselect.repository.transition_fsm import TransitionFSM


def test_add_transition() -> None:
    """Test adding a transition to the transition FSM"""

    # 0 -> 1 -> 0 -> 2 -> 0 -> 1
    # Adjacency List
    # 0 : {1: 2, 2: 1}
    # 1 : {0: 1}
    # 2: {0: 1}
    fsm = TransitionFSM()
    state_ids = [0, 1, 0, 2, 0, 1]
    active_id = state_ids[0]
    for next_id in state_ids[1:]:
        fsm.add_transition(active_id, next_id)
        active_id = next_id

    assert fsm.get_transition_weight(0, 1) == 2
    assert fsm.get_transition_weight(0, 2) == 1
    assert fsm.get_transition_weight(1, 0) == 1
    assert fsm.get_transition_weight(2, 0) == 1
    assert fsm.get_transition_weight(2, 1) == 0
    assert fsm.get_transition_weight(1, 2, smoothing_weight=3) == 3
    assert fsm.get_transition_weight(0, 1, smoothing_weight=3) == 5
