from river.datasets import synth
from river.tree import HoeffdingTreeClassifier

from fall.concept_representations import ErrorRateRepresentation
from fall.repository import Repository
from fall.utils import Observation


def test_step_states() -> None:
    """Test step_all statistics."""
    # pylint: disable="too-many-statements"
    repo = Repository(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=lambda state_id: ErrorRateRepresentation(1, state_id),
    )
    steps = [10, 5, 20]
    s1 = repo.add_next_state()
    active_id = s1.state_id
    assert len(repo.states) == 1
    assert repo.states[s1.state_id] is s1
    assert s1.active_seen_weight == 0
    for _ in range(steps[0]):
        repo.step_all(active_id)
    assert len(repo.states) == 1
    assert repo.states[s1.state_id] is s1
    assert s1.active_seen_weight == steps[0]
    assert s1.seen_weight == steps[0]
    assert s1.weight_since_last_active == 0

    s2 = repo.add_next_state()
    active_id = s2.state_id
    assert len(repo.states) == 2
    assert repo.states[s2.state_id] is s2
    assert s2.active_seen_weight == 0
    for _ in range(steps[1]):
        repo.step_all(active_id)
    assert len(repo.states) == 2
    assert repo.states[s2.state_id] is s2
    assert s1.active_seen_weight == steps[0]
    assert s1.seen_weight == steps[0] + steps[1]
    assert s1.weight_since_last_active == steps[1]
    assert s2.active_seen_weight == steps[1]
    assert s2.seen_weight == steps[1]
    assert s2.weight_since_last_active == 0

    s3 = repo.add_next_state()
    active_id = s3.state_id
    assert len(repo.states) == 3
    assert repo.states[s3.state_id] is s3
    assert s3.active_seen_weight == 0
    for _ in range(steps[2]):
        repo.step_all(active_id)
    assert len(repo.states) == 3
    assert repo.states[s3.state_id] is s3
    assert s1.active_seen_weight == steps[0]
    assert s1.seen_weight == steps[0] + steps[1] + steps[2]
    assert s1.weight_since_last_active == steps[1] + steps[2]
    assert s2.active_seen_weight == steps[1]
    assert s2.seen_weight == steps[1] + steps[2]
    assert s2.weight_since_last_active == steps[2]
    assert s3.active_seen_weight == steps[2]
    assert s3.seen_weight == steps[2]
    assert s3.weight_since_last_active == 0

    active_id = s1.state_id
    for _ in range(steps[0]):
        repo.step_all(active_id)
    assert len(repo.states) == 3
    assert repo.states[s1.state_id] is s1
    assert s1.active_seen_weight == 2 * steps[0]
    assert s1.seen_weight == 2 * steps[0] + steps[1] + steps[2]
    assert s1.weight_since_last_active == 0
    assert s2.active_seen_weight == steps[1]
    assert s2.seen_weight == steps[0] + steps[1] + steps[2]
    assert s2.weight_since_last_active == steps[2] + steps[0]
    assert s3.active_seen_weight == steps[2]
    assert s3.seen_weight == steps[2] + steps[0]
    assert s3.weight_since_last_active == steps[0]

    assert len(repo.states) == 3
    assert repo.states[s1.state_id] is s1
    assert repo.states[s2.state_id] is s2
    assert repo.states[s3.state_id] is s3

    assert len(repo.base_transitions.adjacency_list) == 0


def test_state_predictions_active() -> None:
    """Test predictions in active mode"""
    # pylint: disable="too-many-statements"
    repo = Repository(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=lambda state_id: ErrorRateRepresentation(1, state_id),
    )
    dataset = synth.STAGGER()
    s1 = repo.add_next_state()
    active_id = s1.state_id
    s1_test_classifier = HoeffdingTreeClassifier()
    for t, (x, y) in enumerate(dataset.take(25)):
        ob = Observation(x, y, t, active_id)
        state_p = repo.get_repository_predictions(ob, "active")
        pt = s1_test_classifier.predict_one(x)
        assert state_p[active_id] == pt
        repo.states[active_id].learn_one(ob)
        s1_test_classifier.learn_one(x, y)
    s2 = repo.add_next_state()
    active_id = s2.state_id
    s2_test_classifier = HoeffdingTreeClassifier()
    for t, (x, y) in enumerate(dataset.take(25), start=25):
        ob = Observation(x, y, t, active_id)
        state_p = repo.get_repository_predictions(ob, "active")
        print(state_p)
        assert len(state_p) == 1
        pt_1 = s1_test_classifier.predict_one(x)
        pt_2 = s2_test_classifier.predict_one(x)
        assert state_p[active_id] == pt_2
        assert repo.states[s1.state_id].predict_one(ob) == pt_1
        repo.states[active_id].learn_one(ob)
        s2_test_classifier.learn_one(x, y)


def test_state_predictions_all() -> None:
    """Test predictions in all mode"""
    # pylint: disable="too-many-statements"
    repo = Repository(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=lambda state_id: ErrorRateRepresentation(1, state_id),
    )
    dataset = synth.STAGGER()
    s1 = repo.add_next_state()
    active_id = s1.state_id
    s1_test_classifier = HoeffdingTreeClassifier()
    for t, (x, y) in enumerate(dataset.take(25)):
        ob = Observation(x, y, t, active_id)
        state_p = repo.get_repository_predictions(ob, "all")
        pt = s1_test_classifier.predict_one(x)
        assert state_p[active_id] == pt
        repo.states[active_id].learn_one(ob)
        s1_test_classifier.learn_one(x, y)
    s2 = repo.add_next_state()
    active_id = s2.state_id
    s2_test_classifier = HoeffdingTreeClassifier()
    for t, (x, y) in enumerate(dataset.take(25), start=25):
        ob = Observation(x, y, t, active_id)
        state_p = repo.get_repository_predictions(ob, "all")
        print(state_p)
        assert len(state_p) == 2
        pt_1 = s1_test_classifier.predict_one(x)
        pt_2 = s2_test_classifier.predict_one(x)
        assert state_p[active_id] == pt_2
        assert state_p[s1.state_id] == pt_1
        repo.states[active_id].learn_one(ob)
        s2_test_classifier.learn_one(x, y)
