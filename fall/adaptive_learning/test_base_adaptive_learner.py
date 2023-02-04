from typing import List, Optional

from river.datasets import synth
from river.drift import ADWIN
from river.tree import HoeffdingTreeClassifier

from fall.adaptive_learning import BaseAdaptiveLearner
from fall.adaptive_learning.reidentification_schedulers import (
    DriftDetectionCheck,
    DriftInfo,
    DriftType,
    PeriodicCheck,
)
from fall.concept_representations import ErrorRateRepresentation, MetaFeatureNormalizer
from fall.repository import AbsoluteValueComparer
from fall.states import State
from fall.utils import Observation


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

    # Test that states get the correct properties
    window_size = 50
    update_period = 50
    al_classifier = BaseAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        representation_window_size=window_size,
        representation_update_period=update_period,
    )

    # Assert background state was constructed
    assert isinstance(al_classifier.get_active_state().classifier, HoeffdingTreeClassifier)
    assert isinstance(al_classifier.get_active_state().get_self_representation(), ErrorRateRepresentation)
    assert al_classifier.get_active_state().get_self_representation().window_size == window_size
    assert al_classifier.get_active_state().get_self_representation().update_period == update_period
    # Check that states are correctly made as the concept mode
    assert al_classifier.get_active_state().get_self_representation().mode == "concept"


def test_base_predictions() -> None:
    """Test predictions are the same as made by a base classifier."""
    al_classifier = BaseAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
    )

    normalizer = MetaFeatureNormalizer()
    baseline_state = State(
        HoeffdingTreeClassifier(),
        lambda state_id: ErrorRateRepresentation(al_classifier.representation_window_size, state_id, normalizer),
        state_id=-1,
    )
    baseline_active_representation = ErrorRateRepresentation(
        al_classifier.representation_window_size, baseline_state.state_id, normalizer
    )
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
        _ = baseline_detector.update(
            baseline_comparer.get_state_rep_similarity(baseline_state, baseline_active_representation)  # type: ignore
        )

        in_drift = baseline_detector.drift_detected
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
    normalizer = MetaFeatureNormalizer()
    baseline_state = State(
        HoeffdingTreeClassifier(),
        lambda state_id: ErrorRateRepresentation(
            al_classifier.representation_window_size, state_id, normalizer, mode="concept"
        ),
        state_id=-1,
    )
    baseline_active_representation = ErrorRateRepresentation(
        al_classifier.representation_window_size, baseline_state.state_id, normalizer
    )
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
        assert baseline_relevance == al_classifier.performance_monitor.active_state_last_relevance
        assert baseline_relevance == al_classifier.performance_monitor.background_state_relevance
        _ = baseline_detector.update(baseline_relevance)  # type: ignore

        assert baseline_detector.total == al_classifier.drift_detector.total  # type: ignore

        # We shouldn't find a drift in stable data
        assert not found_drift
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
            assert baseline_relevance == al_classifier.performance_monitor.active_state_last_relevance
            _ = baseline_detector.update(baseline_relevance)  # type: ignore
            in_drift = baseline_detector.drift_detected
            if in_drift:
                found_drift = True
                break

            assert not al_classifier.performance_monitor.in_drift
            assert not al_classifier.performance_monitor.made_transition

    # We should have found a drift when the concept changed
    assert al_classifier.performance_monitor.in_drift
    # background should have been reset since we are using "drift_reset"
    assert al_classifier.background_state is not None
    assert al_classifier.background_state.seen_weight == 0.0
    assert al_classifier.get_active_state().seen_weight == 0.0
    assert len(al_classifier.repository.states) == 2
    assert al_classifier.active_state_id == 1


def test_drift_transition() -> None:
    """Test data after a drift is handled correctly."""
    al_classifier = BaseAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
    )
    normalizer = MetaFeatureNormalizer()
    baseline_c1_state = State(
        HoeffdingTreeClassifier(),
        lambda state_id: ErrorRateRepresentation(
            al_classifier.representation_window_size, state_id, normalizer, mode="concept"
        ),
        state_id=-1,
    )
    baseline_c1_active_representation = ErrorRateRepresentation(
        al_classifier.representation_window_size,
        baseline_c1_state.state_id,
        normalizer,
    )
    baseline_c1_comparer = AbsoluteValueComparer()
    baseline_c1_detector = ADWIN()

    dataset_1 = synth.STAGGER(classification_function=0, seed=0)
    dataset_2 = synth.STAGGER(classification_function=1, seed=0)
    found_drift = False
    drift_point = None
    # Concept 1
    for t, (x, y) in enumerate(dataset_1.take(500)):
        ob = Observation(x=x, y=y, seen_at=t, active_state_id=baseline_c1_state.state_id)
        al_classifier.predict_one(x)
        baseline_c1_state.predict_one(ob, force_train_own_representation=True)
        baseline_c1_active_representation.predict_one(ob)

        al_classifier.learn_one(x, y)
        baseline_c1_state.learn_one(ob, force_train_classifier=True)
        baseline_c1_active_representation.learn_one(ob)
        baseline_c1_relevance = baseline_c1_comparer.get_state_rep_similarity(
            baseline_c1_state, baseline_c1_active_representation
        )
        _ = baseline_c1_detector.update(baseline_c1_relevance)  # type: ignore
        assert not found_drift
        assert not al_classifier.performance_monitor.in_drift
        assert not al_classifier.performance_monitor.made_transition

    # Concept 2
    for t, (x, y) in enumerate(dataset_2.take(500), start=500):
        ob = Observation(x=x, y=y, seen_at=t, active_state_id=baseline_c1_state.state_id)
        al_classifier.predict_one(x)
        baseline_c1_state.predict_one(ob, force_train_own_representation=True)
        baseline_c1_active_representation.predict_one(ob)

        al_classifier.learn_one(x, y)
        baseline_c1_state.learn_one(ob, force_train_classifier=True)
        baseline_c1_active_representation.learn_one(ob)

        baseline_c1_relevance = baseline_c1_comparer.get_state_rep_similarity(
            baseline_c1_state, baseline_c1_active_representation
        )
        assert baseline_c1_relevance == al_classifier.performance_monitor.active_state_last_relevance
        _ = baseline_c1_detector.update(baseline_c1_relevance)  # type: ignore
        in_drift = baseline_c1_detector.drift_detected
        if in_drift:
            found_drift = True
            drift_point = t
            break

        assert not al_classifier.performance_monitor.in_drift
        assert not al_classifier.performance_monitor.made_transition

    # We should have found a drift when the concept changed
    assert al_classifier.performance_monitor.in_drift
    # background should have been reset since we are using "drift_reset"
    assert al_classifier.background_state is not None
    assert al_classifier.background_state.seen_weight == 0.0
    assert al_classifier.get_active_state().seen_weight == 0.0
    assert len(al_classifier.repository.states) == 2
    assert al_classifier.active_state_id == 1
    assert drift_point

    normalizer = MetaFeatureNormalizer()
    # Test that after the transition, we are properly using the new state not the old state.
    baseline_c2_state = State(
        HoeffdingTreeClassifier(),
        lambda state_id: ErrorRateRepresentation(
            al_classifier.representation_window_size, state_id, normalizer, mode="concept"
        ),
        state_id=-2,
    )
    baseline_c2_active_representation = ErrorRateRepresentation(
        al_classifier.representation_window_size, baseline_c2_state.state_id, normalizer
    )
    baseline_c2_comparer = AbsoluteValueComparer()
    baseline_c2_detector = ADWIN()
    # Concept 2
    for t, (x, y) in enumerate(dataset_2.take(500), start=500 + drift_point):
        ob = Observation(x=x, y=y, seen_at=t, active_state_id=baseline_c2_state.state_id)
        assert al_classifier.active_state_id == 1
        p_c2 = al_classifier.predict_one(x)
        bp_c2 = baseline_c2_state.predict_one(ob, force_train_own_representation=True)
        # the adaptive learner should give the same results as a new classifier trained on the new concept.
        assert p_c2 == bp_c2
        # The original concept 1 state should be stored, and give the same predictions as the baseline trained
        # only on that data.
        p_c1 = al_classifier.repository.states[0].predict_one(ob)
        bp_c1 = baseline_c1_state.predict_one(ob)
        assert p_c1 == bp_c1
        baseline_c2_active_representation.predict_one(ob)

        al_classifier.learn_one(x, y)
        baseline_c2_state.learn_one(ob, force_train_classifier=True)
        baseline_c2_active_representation.learn_one(ob)

        baseline_c2_relevance = baseline_c2_comparer.get_state_rep_similarity(
            baseline_c2_state, baseline_c2_active_representation
        )
        assert baseline_c2_relevance == al_classifier.performance_monitor.active_state_last_relevance
        _ = baseline_c2_detector.update(baseline_c2_relevance)  # type: ignore
        in_drift = baseline_c2_detector.drift_detected
        if in_drift:
            found_drift = True
            drift_point = t
            break

        assert not al_classifier.performance_monitor.in_drift
        assert not al_classifier.performance_monitor.made_transition


def test_reidentification_schedule_detection() -> None:
    """Test that drifts are scheduled at the correct times using the DriftDetectionScheduler."""
    # In this case, we want to see a reidentification check performed 50 timesteps after every drift.
    check_delay = 50
    classifier = BaseAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
        reidentification_check_schedulers=[DriftDetectionCheck(check_delay)],
        representation_window_size=50,
    )

    dataset_0 = synth.STAGGER(classification_function=0, seed=0)
    dataset_1 = synth.STAGGER(classification_function=1, seed=0)
    dataset_2 = synth.STAGGER(classification_function=2, seed=0)
    active_state_segments: List[Optional[int]] = [None]
    drift_checks: List[Optional[DriftInfo]] = [None]
    t = 0
    for dataset in [dataset_0, dataset_1, dataset_2] * 3:
        for x, y in dataset.take(500):
            _ = classifier.predict_one(x, t)
            classifier.learn_one(x, y, timestep=t)
            current_id = classifier.performance_monitor.final_active_state_id
            current_drift = classifier.performance_monitor.last_drift
            if current_id != active_state_segments[-1]:
                active_state_segments.append(current_id)
            if current_drift != drift_checks[-1]:
                drift_checks.append(current_drift)
            t += 1

    for i, drift in enumerate(drift_checks):
        if drift is None:
            continue
        if drift.drift_type == DriftType.ScheduledOne:
            prev_drift = drift_checks[i - 1]
            assert prev_drift is not None
            assert prev_drift.drift_type == DriftType.DriftDetectorTriggered or prev_drift.triggered_transition
            assert prev_drift.drift_timestep == drift.drift_timestep - check_delay - 1


def test_reidentification_schedule_periodic() -> None:
    """Test that drifts are scheduled at the correct times using the PeriodicCheck."""
    # In this case, we want to see a reidentification check performed every 50.
    check_period = 100
    classifier = BaseAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
        reidentification_check_schedulers=[PeriodicCheck(check_period)],
        representation_window_size=50,
    )

    dataset_0 = synth.STAGGER(classification_function=0, seed=0)
    dataset_1 = synth.STAGGER(classification_function=1, seed=0)
    dataset_2 = synth.STAGGER(classification_function=2, seed=0)
    active_state_segments: List[Optional[int]] = [None]
    drift_checks: List[Optional[DriftInfo]] = [None]
    t = 0
    for dataset in [dataset_0, dataset_1, dataset_2] * 3:
        for x, y in dataset.take(500):
            _ = classifier.predict_one(x, t)
            classifier.learn_one(x, y, timestep=t)
            current_id = classifier.performance_monitor.final_active_state_id
            current_drift = classifier.performance_monitor.last_drift
            if current_id != active_state_segments[-1]:
                active_state_segments.append(current_id)
            if current_drift != drift_checks[-1]:
                drift_checks.append(current_drift)
            t += 1

    for i, drift in enumerate(drift_checks):
        if drift is None:
            continue
        if drift.drift_type == DriftType.ScheduledOne:
            prev_drift = drift_checks[i - 1]
            print(drift, prev_drift)
            assert prev_drift is not None
            if prev_drift.triggered_transition:
                assert prev_drift.drift_timestep == drift.drift_timestep - check_period - 1
            else:
                assert prev_drift.drift_timestep == drift.drift_timestep - check_period


# %%
