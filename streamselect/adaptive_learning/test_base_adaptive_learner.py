from river import synth
from river.drift import ADWIN
from river.tree import HoeffdingTreeClassifier

from streamselect.adaptive_learning import BaseAdaptiveLearner
from streamselect.concept_representations import ErrorRateRepresentation
from streamselect.repository import AbsoluteValueComparer
from streamselect.states import State
from streamselect.utils import Observation


# pylint: disable=too-many-statements, duplicate-code, R0801
def test_init() -> None:
    """Test initialization of the base class."""
    al_classifier = BaseAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
    )

    # Check initial state has been constructed
    assert len(al_classifier.repository.states) == 1
    assert al_classifier.active_state_id in al_classifier.repository.states
    assert al_classifier.active_state_id in al_classifier.active_window_state_representations

    # Assert background state was constructed
    assert al_classifier.background_state
    assert al_classifier.background_state_active_representation
    assert al_classifier.background_state_detector

    al_classifier = BaseAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="transition_reset",
    )

    # Assert background state was constructed
    assert al_classifier.background_state
    assert al_classifier.background_state_active_representation
    assert not al_classifier.background_state_detector

    al_classifier = BaseAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="transition_reset",
    )

    # Assert background state was constructed
    assert al_classifier.background_state
    assert al_classifier.background_state_active_representation
    assert not al_classifier.background_state_detector

    al_classifier = BaseAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode=None,
    )

    # Assert background state was constructed
    assert not al_classifier.background_state
    assert not al_classifier.background_state_active_representation
    assert not al_classifier.background_state_detector


def test_base_predictions() -> None:
    """Test predictions are the same as made by a base classifier."""
    al_classifier = BaseAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
    )

    baseline_state = State(
        HoeffdingTreeClassifier(), lambda state_id: ErrorRateRepresentation(1, state_id), state_id=-1
    )
    baseline_active_representation = ErrorRateRepresentation(1, baseline_state.state_id)
    baseline_comparer = AbsoluteValueComparer()
    baseline_detector = ADWIN()

    dataset = synth.STAGGER()
    for t, (x, y) in enumerate(dataset.take(50)):
        p = al_classifier.predict_one(x, t)
        ob = Observation(x=x, y=y, seen_at=t, active_state_id=baseline_state.state_id)
        p_b = baseline_state.predict_one(ob)
        baseline_active_representation.predict_one(ob)
        assert p == p_b

        al_classifier.learn_one(x, y, timestep=t)
        baseline_state.learn_one(ob)
        p_b = baseline_state.predict_one(ob)
        baseline_active_representation.learn_one(ob)
        in_drift, _ = baseline_detector.update(
            baseline_comparer.get_state_rep_similarity(baseline_state, baseline_active_representation)  # type: ignore
        )

        if in_drift:
            break

        assert not al_classifier.performance_monitor.in_drift
        assert not al_classifier.performance_monitor.made_transition


def test_drift_detection() -> None:
    """Test predictions are the same as made by a base classifier, and drift detection capabilities are as well."""
    al_classifier = BaseAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
    )

    baseline_state = State(
        HoeffdingTreeClassifier(), lambda state_id: ErrorRateRepresentation(1, state_id, mode="concept"), state_id=-1
    )
    baseline_active_representation = ErrorRateRepresentation(1, baseline_state.state_id)
    baseline_comparer = AbsoluteValueComparer()
    baseline_detector = ADWIN()

    dataset_0 = synth.STAGGER(classification_function=0, seed=0)
    dataset_1 = synth.STAGGER(classification_function=1, seed=0)
    found_drift = False
    for t, (x, y) in enumerate(dataset_0.take(500)):
        # Ensure predictions are equal
        ob = Observation(x=x, y=y, seen_at=t, active_state_id=baseline_state.state_id)
        p = al_classifier.predict_one(x)
        p_b = baseline_state.predict_one(ob, force_train_own_representation=True)
        baseline_active_representation.predict_one(ob)
        # Ensure background predictions are equal, since we are using drift_reset and no drift will occur.
        p_background = al_classifier.background_state.predict_one(ob)  # type: ignore
        assert al_classifier.background_state
        assert al_classifier.background_state_active_representation
        assert al_classifier.background_state_detector
        assert p_b == p_background
        assert p == p_b
        assert (
            baseline_active_representation.meta_feature_values[0]
            == al_classifier.active_window_state_representations[al_classifier.active_state_id].meta_feature_values[0]
        )
        assert (
            baseline_active_representation.meta_feature_values[0]
            == al_classifier.background_state_active_representation.meta_feature_values[0]
        )

        # Assert learning and relevance checks are equal.
        # Note: we have to use the second prediction from the baseline, as for the very
        # first prediction in the stream the first prediction is None as classes haven't been
        # learned. We do this automatically in the adaptive_learning class.
        al_classifier.learn_one(x, y)
        baseline_state.learn_one(ob, force_train_classifier=True)
        baseline_active_representation.learn_one(ob)
        baseline_relevance = baseline_comparer.get_state_rep_similarity(baseline_state, baseline_active_representation)
        assert (
            baseline_state.get_self_representation().meta_feature_values[0]
            == al_classifier.get_active_state().get_self_representation().meta_feature_values[0]
        )
        assert (
            baseline_state.get_self_representation().meta_feature_values[0]
            == al_classifier.background_state.get_self_representation().meta_feature_values[0]
        )
        assert (
            baseline_active_representation.meta_feature_values[0]
            == al_classifier.active_window_state_representations[al_classifier.active_state_id].meta_feature_values[0]
        )
        assert (
            baseline_active_representation.meta_feature_values[0]
            == al_classifier.background_state_active_representation.meta_feature_values[0]
        )
        assert baseline_relevance == al_classifier.performance_monitor.active_state_relevance
        assert baseline_relevance == al_classifier.performance_monitor.background_state_relevance
        in_drift, _ = baseline_detector.update(baseline_relevance)  # type: ignore

        assert baseline_detector.total == al_classifier.drift_detector.total  # type: ignore

        if in_drift:
            found_drift = True
            break

        assert not al_classifier.performance_monitor.in_drift
        assert not al_classifier.performance_monitor.made_transition
    if not found_drift:
        for t, (x, y) in enumerate(dataset_1.take(500), start=500):
            ob = Observation(x=x, y=y, seen_at=t, active_state_id=baseline_state.state_id)
            p = al_classifier.predict_one(x)
            p_b = baseline_state.predict_one(ob, force_train_own_representation=True)
            baseline_active_representation.predict_one(ob)
            assert p == p_b

            al_classifier.learn_one(x, y)
            baseline_state.learn_one(ob, force_train_classifier=True)
            baseline_active_representation.learn_one(ob)
            baseline_relevance = baseline_comparer.get_state_rep_similarity(
                baseline_state, baseline_active_representation
            )
            assert baseline_relevance == al_classifier.performance_monitor.active_state_relevance
            in_drift, _ = baseline_detector.update(baseline_relevance)  # type: ignore

            if in_drift:
                found_drift = True
                break

            assert not al_classifier.performance_monitor.in_drift
            assert not al_classifier.performance_monitor.made_transition

    assert al_classifier.performance_monitor.in_drift == found_drift
    # background should have been reset since we are using "drift_reset"
    assert al_classifier.background_state is not None
    assert al_classifier.background_state.seen_weight == 0.0
