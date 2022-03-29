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


def test_del_transition() -> None:
    """Test deleting a transition from the transition FSM"""

    # 0 -> 1 -> 0 -> 2 -> 0 -> 1 -> 2
    # Adjacency List
    # 0 : {1: 2, 2: 1}
    # 1 : {0: 1, 2: 1}
    # 2: {0: 1}
    fsm = TransitionFSM()
    state_ids = [0, 1, 0, 2, 0, 1, 2]
    active_id = state_ids[0]
    for next_id in state_ids[1:]:
        fsm.add_transition(active_id, next_id)
        active_id = next_id

    fsm.delete_state(1, retain_indirect_weight=False)
    # Adjacency List
    # 0 : {2: 1}
    # 2: {0: 1}

    assert fsm.get_transition_weight(0, 2) == 1
    assert fsm.get_transition_weight(2, 0) == 1
    assert fsm.get_transition_weight(0, 1) == 0
    assert fsm.get_transition_weight(1, 2) == 0
    assert fsm.in_degrees[0].direct_weight == 1
    assert fsm.in_degrees[0].indirect_weight == 0
    assert fsm.in_degrees[0].total_weight == 1
    assert fsm.in_degrees[2].direct_weight == 1
    assert fsm.in_degrees[2].indirect_weight == 0
    assert fsm.in_degrees[2].total_weight == 1
    assert 1 not in fsm.in_degrees
    assert 1 not in fsm.out_degrees
    assert 1 not in fsm.adjacency_list


def test_del_retain_transition() -> None:
    """Test deleting a transition, but maintaining
    indirect weights for 1 step."""
    # 0 -> 1 -> 0 -> 2 -> 0 -> 1 -> 2
    # Adjacency List
    # 0 : {1: 2, 2: 1}
    # 1 : {0: 1, 2: 1}
    # 2: {0: 1}
    fsm = TransitionFSM()
    state_ids = [0, 1, 0, 2, 0, 1, 2]
    active_id = state_ids[0]
    for next_id in state_ids[1:]:
        fsm.add_transition(active_id, next_id)
        active_id = next_id

    fsm.delete_state(1, retain_indirect_weight=True)
    # Adjacency List
    # 0 : {2: 2 (1d, 1id), 0: 1 (1id)}
    # 2: {0: 1}

    assert fsm.get_transition_weight(0, 2) == 2
    assert fsm.adjacency_list[0][2].direct_weight == 1
    assert fsm.adjacency_list[0][2].indirect_weight == 1
    assert fsm.adjacency_list[0][2].total_weight == 2
    assert fsm.get_transition_weight(0, 0) == 1
    assert fsm.adjacency_list[0][0].direct_weight == 0
    assert fsm.adjacency_list[0][0].indirect_weight == 1
    assert fsm.adjacency_list[0][0].total_weight == 1
    assert fsm.get_transition_weight(2, 0) == 1
    assert fsm.get_transition_weight(0, 1) == 0
    assert fsm.get_transition_weight(1, 2) == 0
    assert 1 not in fsm.in_degrees
    assert 1 not in fsm.out_degrees
    assert 1 not in fsm.adjacency_list
