from river import synth
from river.tree import HoeffdingTreeClassifier

from streamselect.concept_representations import ErrorRateRepresentation
from streamselect.states import State


def test_classifier_equivalence() -> None:
    """Calling predict_one and learn_one on a state
    should be equivalent to calling directly on the underlying state."""

    state = State(HoeffdingTreeClassifier(), lambda: ErrorRateRepresentation(1))
    base_classifier = HoeffdingTreeClassifier()
    preds = []
    for x, y in synth.STAGGER().take(100):
        s_p = state.predict_one(x)
        bc_p = base_classifier.predict_one(x)
        preds.append((s_p, bc_p))
        state.learn_one(x, y)
        base_classifier.learn_one(x, y)

    assert all(p[0] == p[1] for p in preds)


def test_representation_equivalence() -> None:
    """Calling predict_one and learn_one on a state
    should be equivalent to calling directly on the underlying concept_representaion."""

    state = State(HoeffdingTreeClassifier(), lambda: ErrorRateRepresentation(5))
    base_representation = ErrorRateRepresentation(5)
    preds = []
    for x, y in synth.STAGGER().take(100):
        p = state.predict_one(x)
        base_representation.predict_one(x, p)
        preds.append((state.get_self_representation().get_values(), base_representation.get_values()))
        state.learn_one(x, y)
        base_representation.learn_one(x, y, p)
    print(preds)
    assert all(p[0] == p[1] for p in preds)


def test_representation_notrain() -> None:
    """Calling predict_one and learn_one on a state
    should not update the representation if disabled."""
    # Test that we can dynamically disable
    state = State(HoeffdingTreeClassifier(), lambda: ErrorRateRepresentation(5))
    base_representation = ErrorRateRepresentation(5)
    preds = []
    disable_rep = False
    for i, (x, y) in enumerate(synth.STAGGER().take(100)):
        p = state.predict_one(x)
        preds.append((state.get_self_representation().get_values(), base_representation.get_values()))
        state.learn_one(x, y)
        if not disable_rep:
            base_representation.predict_one(x, p)
            base_representation.learn_one(x, y, p)
        if i == 50:
            state.deactivate_train_representation()
            disable_rep = True
    print(preds)
    assert all(p[0] == p[1] for p in preds)

    # Test that we can disable at start
    state = State(HoeffdingTreeClassifier(), lambda: ErrorRateRepresentation(5))
    base_representation = ErrorRateRepresentation(5)
    preds = []
    disable_rep = True
    state.deactivate_train_representation()
    for i, (x, y) in enumerate(synth.STAGGER().take(100)):
        p = state.predict_one(x)
        preds.append((state.get_self_representation().get_values(), base_representation.get_values()))
        state.learn_one(x, y)
        if not disable_rep:
            base_representation.predict_one(x, p)
            base_representation.learn_one(x, y, p)
    print(preds)
    assert all(p[0] == p[1] for p in preds)
