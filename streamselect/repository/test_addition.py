from river.tree import HoeffdingTreeClassifier

from streamselect.concept_representations import ErrorRateRepresentation
from streamselect.repository import Repository


def test_add_next_state() -> None:
    """Test automatic repo state construction."""
    repo = Repository(
        classifier_constructor=HoeffdingTreeClassifier, representation_constructor=lambda: ErrorRateRepresentation(1)
    )

    s1 = repo.add_next_state()
    s2 = repo.add_next_state()
    s3 = repo.add_next_state()

    assert len(repo.states) == 3
    assert repo.states[s1.state_id] is s1
    assert repo.states[s2.state_id] is s2
    assert repo.states[s3.state_id] is s3

    assert len(repo.base_transitions.adjacency_list) == 0

    s1 = repo.add_next_state()
    active = s1
    s2 = repo.add_next_state()
    repo.add_transition(active, s2)
    active = s2
    s3 = repo.add_next_state()
    repo.add_transition(active, s3)
    active = s3

    assert len(repo.states) == 6
    assert repo.states[s1.state_id] is s1
    assert repo.states[s2.state_id] is s2
    assert repo.states[s3.state_id] is s3

    print(repo.base_transitions.adjacency_list)
    assert len(repo.base_transitions.adjacency_list) == 3
