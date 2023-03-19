from river.datasets import synth
from river.tree import HoeffdingTreeClassifier

from fall.concept_representations import ErrorRateRepresentation, MetaFeatureNormalizer
from fall.states import State
from fall.utils import Observation


def test_classifier_equivalence() -> None:
    """Calling predict_one and learn_one on a state
    should be equivalent to calling directly on the underlying state."""
    normalizer = MetaFeatureNormalizer()
    state = State(HoeffdingTreeClassifier(), lambda state_id: ErrorRateRepresentation(1, state_id, normalizer))
    base_classifier = HoeffdingTreeClassifier()
    preds = []
    for t, (x, y) in enumerate(synth.STAGGER().take(100)):
        ob = Observation(x, y, t, state.state_id)
        s_p = state.predict_one(ob)
        ob_p = ob.predictions[state.state_id]
        bc_p = base_classifier.predict_one(x)
        preds.append((s_p, bc_p, ob_p))
        state.learn_one(ob)
        base_classifier.learn_one(x, y)

    assert all(p[0] == p[1] == p[2] for p in preds)


def test_classifier_inactive() -> None:
    """Calling predict_one and learn_one on a state
    should be equivalent to calling directly on the underlying state.
    Since the state is not active, the classifier should not train."""
    normalizer = MetaFeatureNormalizer()
    state = State(
        HoeffdingTreeClassifier(), lambda state_id: ErrorRateRepresentation(1, state_id, normalizer), state_id=1
    )
    base_classifier = HoeffdingTreeClassifier()
    preds = []
    for t, (x, y) in enumerate(synth.STAGGER().take(100)):
        # Test an observation with a different active state_id
        ob = Observation(x, y, t, -1)
        s_p = state.predict_one(ob)
        ob_p = ob.predictions[state.state_id]
        bc_p = base_classifier.predict_one(x)
        preds.append((s_p, bc_p, ob_p))
        state.learn_one(ob)

        # We do not train the base classifier, which should match the state classifier
        # as the state is not active.

    assert all(p[0] == p[1] == p[2] for p in preds)


def test_classifier_inactive_force() -> None:
    """Calling predict_one and learn_one on a state
    should be equivalent to calling directly on the underlying state.
    Since the state is not active, but training is forced, the classifier should train."""
    normalizer = MetaFeatureNormalizer()
    state = State(
        HoeffdingTreeClassifier(), lambda state_id: ErrorRateRepresentation(1, state_id, normalizer), state_id=1
    )
    base_classifier = HoeffdingTreeClassifier()
    preds = []
    for t, (x, y) in enumerate(synth.STAGGER().take(100)):
        # Test an observation with a different active state_id
        ob = Observation(x, y, t, -1)
        s_p = state.predict_one(ob)
        ob_p = ob.predictions[state.state_id]
        bc_p = base_classifier.predict_one(x)
        preds.append((s_p, bc_p, ob_p))
        state.learn_one(ob, force_train_classifier=True)
        base_classifier.learn_one(x, y)

        # We train the base classifier, which should match the state classifier
        # as the state is not active but training is forced.

    assert all(p[0] == p[1] == p[2] for p in preds)


def test_representation_equivalence() -> None:
    """Calling predict_one and learn_one on a state
    should be equivalent to calling directly on the underlying concept_representaion."""
    normalizer = MetaFeatureNormalizer()
    state = State(HoeffdingTreeClassifier(), lambda state_id: ErrorRateRepresentation(5, state_id, normalizer))
    base_representation = ErrorRateRepresentation(5, 5, normalizer)
    preds = []
    for t, (x, y) in enumerate(synth.STAGGER().take(100)):
        ob = Observation(x, y, t, state.state_id)
        p = state.predict_one(ob)
        ob.add_prediction(p, 5)
        base_representation.predict_one(ob)
        preds.append((state.get_self_representation().get_values(), base_representation.get_values()))
        state.learn_one(ob)
        base_representation.learn_one(ob)
    print(preds)
    assert all(p[0] == p[1] for p in preds)


def test_representation_equivalence_inactive() -> None:
    """Calling predict_one and learn_one on a state when it is unactive.
    should be equivalent to calling directly on the underlying concept_representaion."""
    normalizer = MetaFeatureNormalizer()
    state = State(
        HoeffdingTreeClassifier(), lambda state_id: ErrorRateRepresentation(5, state_id, normalizer), state_id=1
    )
    base_representation = ErrorRateRepresentation(5, 5, normalizer)
    base_representation_trained = ErrorRateRepresentation(5, 5, normalizer)
    preds = []
    for t, (x, y) in enumerate(synth.STAGGER().take(100)):
        active_state_id = -1
        ob = Observation(x, y, t, active_state_id)
        p = state.predict_one(ob)
        ob.add_prediction(p, 5)
        base_representation.predict_one(ob)
        base_representation_trained.predict_one(ob)
        preds.append(
            (
                state.get_self_representation().get_values(),
                base_representation.get_values(),
                state.concept_representation[active_state_id].get_values(),
                base_representation_trained.get_values(),
            )
        )
        state.learn_one(ob)
        base_representation_trained.learn_one(ob)
    assert all((p[0] == p[1]) and (p[2] == p[3]) for p in preds)


def test_representation_equivalence_inactive_forced() -> None:
    """Calling predict_one and learn_one on a state when it is unactive.
    should be equivalent to calling directly on the underlying concept_representaion."""
    normalizer = MetaFeatureNormalizer()
    state = State(
        HoeffdingTreeClassifier(), lambda state_id: ErrorRateRepresentation(5, state_id, normalizer), state_id=1
    )
    base_representation = ErrorRateRepresentation(5, 5, normalizer)
    base_representation_trained = ErrorRateRepresentation(5, 5, normalizer)
    preds = []
    for t, (x, y) in enumerate(synth.STAGGER().take(100)):
        active_state_id = -1
        ob = Observation(x, y, t, active_state_id)
        p = state.predict_one(ob, force_train_own_representation=True)
        ob.add_prediction(p, 5)
        base_representation.predict_one(ob)
        base_representation_trained.predict_one(ob)
        preds.append(
            (
                state.get_self_representation().get_values(),
                base_representation.get_values(),
                None,
                base_representation_trained.get_values(),
            )
        )
        state.learn_one(ob, force_train_classifier=True)
        base_representation_trained.learn_one(ob)
        assert active_state_id not in state.concept_representation
    print(preds)
    assert all(p[0] == p[3] for p in preds)


def test_representation_notrain() -> None:
    """Calling predict_one and learn_one on a state
    should not update the representation if disabled."""
    # Test that we can dynamically disable
    normalizer = MetaFeatureNormalizer()
    state = State(HoeffdingTreeClassifier(), lambda state_id: ErrorRateRepresentation(5, state_id, normalizer))
    base_representation = ErrorRateRepresentation(5, 5, normalizer)
    preds = []
    disable_rep = False
    for i, (x, y) in enumerate(synth.STAGGER().take(100)):
        ob = Observation(x, y, i, state.state_id)
        p = state.predict_one(ob)
        ob.add_prediction(p, 5)
        preds.append((state.get_self_representation().get_values(), base_representation.get_values()))
        state.learn_one(ob)
        if not disable_rep:
            base_representation.predict_one(ob)
            base_representation.learn_one(ob)
        if i == 50:
            state.deactivate_train_representation()
            disable_rep = True
    print(preds)
    assert all(p[0] == p[1] for p in preds)

    # Test that we can disable at start
    normalizer = MetaFeatureNormalizer()
    state = State(HoeffdingTreeClassifier(), lambda state_id: ErrorRateRepresentation(5, state_id, normalizer))
    base_representation = ErrorRateRepresentation(5, 5, normalizer)
    preds = []
    disable_rep = True
    state.deactivate_train_representation()
    for i, (x, y) in enumerate(synth.STAGGER().take(100)):
        ob = Observation(x, y, i, state.state_id)
        p = state.predict_one(ob)
        ob.add_prediction(p, 5)
        preds.append((state.get_self_representation().get_values(), base_representation.get_values()))
        state.learn_one(ob)
        if not disable_rep:
            base_representation.predict_one(ob)
            base_representation.learn_one(ob)
    print(preds)
    assert all(p[0] == p[1] for p in preds)
