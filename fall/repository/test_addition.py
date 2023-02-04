from river.tree import HoeffdingTreeClassifier

from fall.concept_representations import ErrorRateRepresentation, MetaFeatureNormalizer
from fall.repository import Repository


def test_make_state() -> None:
    """Test the make state function."""
    normalizer = MetaFeatureNormalizer()
    repo = Repository(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=lambda state_id: ErrorRateRepresentation(1, state_id, normalizer),
    )
    s1 = repo.make_state(1)
    s2 = repo.make_state(2)
    s3 = repo.make_state(-1)
    assert s1 is not s2
    assert s1.classifier is not s2.classifier
    assert s1.concept_representation is not s2.concept_representation
    assert s2 is not s3
    assert s2.classifier is not s3.classifier
    assert s2.concept_representation is not s3.concept_representation


def test_add_next_state() -> None:
    """Test automatic repo state construction."""
    normalizer = MetaFeatureNormalizer()
    repo = Repository(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=lambda state_id: ErrorRateRepresentation(1, state_id, normalizer),
    )
    # pylint: disable=too-many-statements, duplicate-code, R0801
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
