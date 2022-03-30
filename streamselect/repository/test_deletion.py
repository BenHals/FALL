from river.tree import HoeffdingTreeClassifier

from streamselect.concept_representations import ErrorRateRepresentation
from streamselect.repository import Repository, ValuationPolicy


def test_fifo_policy() -> None:
    """Test FIFO policy, should delete oldest state"""
    repo = Repository(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=lambda: ErrorRateRepresentation(1),
        valuation_policy=ValuationPolicy.FIFO,
    )
    s1 = repo.add_next_state()
    active = s1
    for _ in range(100):
        repo.step_all(active.state_id)
    s2 = repo.add_next_state()
    repo.add_transition(active, s2)
    active = s2
    for _ in range(100):
        repo.step_all(active.state_id)
    active = s1
    for _ in range(100):
        repo.step_all(active.state_id)
    s3 = repo.add_next_state()
    repo.add_transition(active, s3)
    active = s3
    for _ in range(100):
        repo.step_all(active.state_id)

    assert len(repo.states) == 3
    assert repo.states[s1.state_id].seen_weight == 400
    assert repo.states[s1.state_id].active_seen_weight == 200
    assert repo.states[s1.state_id].weight_since_last_active == 100
    assert repo.states[s2.state_id].seen_weight == 300
    assert repo.states[s2.state_id].active_seen_weight == 100
    assert repo.states[s2.state_id].weight_since_last_active == 200
    assert repo.states[s3.state_id].seen_weight == 100
    assert repo.states[s3.state_id].active_seen_weight == 100
    assert repo.states[s3.state_id].weight_since_last_active == 0

    repo.memory_management_deletion()
    assert len(repo.states) == 2
    assert s1.state_id not in repo.states


def test_lru_policy() -> None:
    """Test LRU policy, should delete least recently used state"""
    repo = Repository(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=lambda: ErrorRateRepresentation(1),
        valuation_policy=ValuationPolicy.LRU,
    )
    s1 = repo.add_next_state()
    active = s1
    for _ in range(100):
        repo.step_all(active.state_id)
    s2 = repo.add_next_state()
    repo.add_transition(active, s2)
    active = s2
    for _ in range(100):
        repo.step_all(active.state_id)
    active = s1
    for _ in range(100):
        repo.step_all(active.state_id)
    s3 = repo.add_next_state()
    repo.add_transition(active, s3)
    active = s3
    for _ in range(100):
        repo.step_all(active.state_id)

    repo.memory_management_deletion()
    assert len(repo.states) == 2
    assert s2.state_id not in repo.states
