from typing import List, Optional

import numpy as np
from pytest import approx
from river.datasets import synth
from river.drift import ADWIN
from river.tree import HoeffdingTreeClassifier

from fall.adaptive_learning import (
    BaseAdaptiveLearner,
    BaseBufferedAdaptiveLearner,
    get_constant_max_buffer_scheduler,
    get_increasing_buffer_scheduler,
)
from fall.adaptive_learning.reidentification_schedulers import (
    DriftDetectionCheck,
    DriftInfo,
    DriftType,
    PeriodicCheck,
)
from fall.concept_representations import ErrorRateRepresentation
from fall.repository import AbsoluteValueComparer


def test_init() -> None:
    """Test initialization correctly initializes the base class."""
    buffer_timeout = 5
    al_classifier = BaseBufferedAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
        buffer_timeout_max=buffer_timeout,
    )

    # Check initial state has been constructed
    assert len(al_classifier.repository.states) == 1
    assert al_classifier.active_state_id in al_classifier.repository.states
    assert al_classifier.active_state_id in al_classifier.active_window_state_representations

    # Assert background state was constructed
    assert al_classifier.background_state
    assert al_classifier.background_state_active_representation
    assert al_classifier.background_state_detector
    assert al_classifier.buffer_timeout_max == buffer_timeout
    # Due to the default increasing scheduler
    assert al_classifier.buffer_timeout == 0.0
    assert al_classifier.buffer.supervised_buffer_timeout == 0.0
    assert al_classifier.buffer.unsupervised_buffer_timeout == 0.0
    buffer_timeout = 5
    al_classifier = BaseBufferedAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
        buffer_timeout_max=buffer_timeout,
        buffer_timeout_scheduler=get_constant_max_buffer_scheduler(),
    )

    # Check initial state has been constructed
    assert len(al_classifier.repository.states) == 1
    assert al_classifier.active_state_id in al_classifier.repository.states
    assert al_classifier.active_state_id in al_classifier.active_window_state_representations

    # Assert background state was constructed
    assert al_classifier.background_state
    assert al_classifier.background_state_active_representation
    assert al_classifier.background_state_detector
    assert al_classifier.buffer_timeout_max == buffer_timeout
    # Due to the constant scheduler
    assert al_classifier.buffer_timeout == buffer_timeout
    assert al_classifier.buffer.supervised_buffer_timeout == buffer_timeout
    assert al_classifier.buffer.unsupervised_buffer_timeout == buffer_timeout

    al_classifier = BaseBufferedAdaptiveLearner(
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

    al_classifier = BaseBufferedAdaptiveLearner(
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

    al_classifier = BaseBufferedAdaptiveLearner(
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
    al_classifier = BaseBufferedAdaptiveLearner(
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
    """Test predictions are the same as made by a base classifier when buffer_timeout is zero"""
    base_classifier = BaseAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
    )
    buffered_classifier = BaseBufferedAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
        buffer_timeout_max=0.0,
    )

    dataset_0 = synth.STAGGER(classification_function=0, seed=0)
    dataset_1 = synth.STAGGER(classification_function=1, seed=0)
    dataset_2 = synth.STAGGER(classification_function=2, seed=0)
    for dataset in [dataset_0, dataset_1, dataset_2] * 3:
        for t, (x, y) in enumerate(dataset.take(500)):
            p_baseline = base_classifier.predict_one(x, t)
            p_buffered = buffered_classifier.predict_one(x, t)
            assert p_baseline == p_buffered

            base_classifier.learn_one(x, y, timestep=t)
            buffered_classifier.learn_one(x, y, timestep=t)


def test_buffer_lag_constant() -> None:
    """Test that each state only learns from observations buffer_timeout prior to the most recent observations.
    Test using the constant scheduler."""
    buffer_timeout = 50
    buffered_classifier = BaseBufferedAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
        buffer_timeout_max=buffer_timeout,
        buffer_timeout_scheduler=get_constant_max_buffer_scheduler(),
    )

    dataset_0 = synth.STAGGER(classification_function=0, seed=0)
    dataset_1 = synth.STAGGER(classification_function=1, seed=0)
    dataset_2 = synth.STAGGER(classification_function=2, seed=0)
    t = 0
    for dataset in [dataset_0, dataset_1, dataset_2] * 3:
        for x, y in dataset.take(500):
            _ = buffered_classifier.predict_one(x, t)
            buffered_classifier.learn_one(x, y, timestep=t)
            assert (
                buffered_classifier.get_active_state().last_trained_active_timestep
                < 0  # At the very start our test will be off due to initialization
                or buffered_classifier.get_active_state().weight_since_last_active
                != 0  # When we transition to a new state will be off until we see an ob
                or (
                    buffered_classifier.get_active_state().weight_since_last_active == 0
                    and buffered_classifier.get_active_state().last_trained_active_timestep != t
                )
                or buffered_classifier.get_active_state().last_trained_active_timestep
                == t - buffer_timeout  # Test that we never train on unbuffered obs
            )

            t += 1


def test_buffer_lag_increasing() -> None:
    """Test that each state only learns from observations buffer_timeout prior to the most recent observations.
    Test using the increasing scheduler."""
    buffer_timeout = 50
    increase_rate = 1.0
    buffered_classifier = BaseBufferedAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
        buffer_timeout_max=buffer_timeout,
        buffer_timeout_scheduler=get_increasing_buffer_scheduler(increase_rate),
    )

    dataset_0 = synth.STAGGER(classification_function=0, seed=0)
    dataset_1 = synth.STAGGER(classification_function=1, seed=0)
    dataset_2 = synth.STAGGER(classification_function=2, seed=0)
    t = 0
    for dataset in [dataset_0, dataset_1, dataset_2] * 3:
        for x, y in dataset.take(500):
            initial_seen_weight = buffered_classifier.get_active_state().active_seen_weight
            _ = buffered_classifier.predict_one(x, t)
            buffered_classifier.learn_one(x, y, timestep=t)
            assert (
                buffered_classifier.get_active_state().last_trained_active_timestep
                < 0  # At the very start our test will be off due to initialization
                or buffered_classifier.get_active_state().weight_since_last_active
                != 0  # When we transition to a new state will be off until we see an ob
                or buffered_classifier.get_active_state().seen_weight
                == 0  # When we transition to a new state will be off until we see an ob
                or (
                    buffered_classifier.get_active_state().weight_since_last_active == 0
                    and buffered_classifier.get_active_state().last_trained_active_timestep != t
                )
                or buffered_classifier.get_active_state().last_trained_active_timestep
                == t
                - min(
                    round(initial_seen_weight * increase_rate), buffer_timeout
                )  # Test that we never train on unbuffered obs
                # This last check tests that each observation we train on has been buffered at least
                # initial_seen_weight * increase_rate observations, up to the buffer_timeout_max.
            )

            t += 1


def test_buffer_lag_increasing_2() -> None:
    """Test that each state only learns from observations buffer_timeout prior to the most recent observations.
    Test using the increasing scheduler. (change increase rate)"""
    buffer_timeout = 50
    increase_rate = 0.1
    buffered_classifier = BaseBufferedAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
        buffer_timeout_max=buffer_timeout,
        buffer_timeout_scheduler=get_increasing_buffer_scheduler(increase_rate),
    )

    dataset_0 = synth.STAGGER(classification_function=0, seed=0)
    dataset_1 = synth.STAGGER(classification_function=1, seed=0)
    dataset_2 = synth.STAGGER(classification_function=2, seed=0)
    t = 0
    for dataset in [dataset_0, dataset_1, dataset_2] * 3:
        for x, y in dataset.take(500):
            initial_seen_weight = buffered_classifier.get_active_state().active_seen_weight
            _ = buffered_classifier.predict_one(x, t)
            buffered_classifier.learn_one(x, y, timestep=t)
            assert (
                buffered_classifier.get_active_state().last_trained_active_timestep
                < 0  # At the very start our test will be off due to initialization
                or buffered_classifier.get_active_state().weight_since_last_active
                != 0  # When we transition to a new state will be off until we see an ob
                or buffered_classifier.get_active_state().seen_weight
                == 0  # When we transition to a new state will be off until we see an ob
                or (
                    buffered_classifier.get_active_state().weight_since_last_active == 0
                    and buffered_classifier.get_active_state().last_trained_active_timestep != t
                )
                or buffered_classifier.get_active_state().last_trained_active_timestep
                == t
                - min(
                    round(initial_seen_weight * increase_rate), buffer_timeout
                )  # Test that we never train on unbuffered obs
                # This last check tests that each observation we train on has been buffered at least
                # initial_seen_weight * increase_rate observations, up to the buffer_timeout_max.
                # We need the round because if an observation is buffered for e.g., 2.6 timesteps,
                # It will not be unbuffered for 3 timesteps.
            )

            t += 1


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


def test_observations_buffered() -> None:
    """Test that the observations worked on in step are the same as in the buffer"""
    buffered_classifier_1 = BaseBufferedAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
        buffer_timeout_max=0.0,
    )
    buffered_classifier_2 = BaseBufferedAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
        buffer_timeout_max=25,
    )

    dataset_0 = synth.STAGGER(classification_function=0, seed=0)
    dataset_1 = synth.STAGGER(classification_function=1, seed=0)
    dataset_2 = synth.STAGGER(classification_function=2, seed=0)
    for dataset in [dataset_0, dataset_1, dataset_2] * 3:
        for t, (x, y) in enumerate(dataset.take(500)):
            _ = buffered_classifier_1.predict_one(x, t)
            _ = buffered_classifier_2.predict_one(x, t)
            buffered_classifier_1.learn_one(x, y, timestep=t)
            buffered_classifier_2.learn_one(x, y, timestep=t)

            # Test this by checking the active_state_relevance, which is set in step
            assert buffered_classifier_1.buffer.supervised_buffer.active_window[-1].active_state_relevance is not None
            assert buffered_classifier_2.buffer.supervised_buffer.active_window[-1].active_state_relevance is not None


def test_is_stable() -> None:
    """Test that observations are not marked stable before the buffer_timeout."""
    buffer_timeout = 25
    buffered_classifier = BaseBufferedAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
        buffer_timeout_max=buffer_timeout,
        buffer_timeout_scheduler=get_constant_max_buffer_scheduler(),
    )

    dataset_0 = synth.STAGGER(classification_function=0, seed=0)
    dataset_1 = synth.STAGGER(classification_function=1, seed=0)
    dataset_2 = synth.STAGGER(classification_function=2, seed=0)
    t = 0
    for dataset in [dataset_0, dataset_1, dataset_2] * 3:
        for (x, y) in dataset.take(500):
            _ = buffered_classifier.predict_one(x, t)
            buffered_classifier.learn_one(x, y, timestep=t)

            for ob in buffered_classifier.buffer.supervised_buffer.buffer:
                assert (ob.seen_at <= t - buffer_timeout) or (ob.is_stable is False)
            for ob in buffered_classifier.buffer.unsupervised_buffer.buffer:
                assert (ob.seen_at <= t - buffer_timeout) or (ob.is_stable is False)
            for ob in buffered_classifier.buffer.supervised_buffer.stable_window:
                assert (ob.seen_at <= t - buffer_timeout) or (ob.is_stable is False)
            for ob in buffered_classifier.buffer.unsupervised_buffer.stable_window:
                assert (ob.seen_at <= t - buffer_timeout) or (ob.is_stable is False)
            for ob in buffered_classifier.supervised_active_window:
                assert (ob.seen_at <= t - buffer_timeout) or (ob.is_stable is False)
            for ob in buffered_classifier.unsupervised_active_window:
                assert (ob.seen_at <= t - buffer_timeout) or (ob.is_stable is False)

            t += 1


def test_base_predictions_increase_rate() -> None:
    """Test predictions are the same as made by a base classifier when buffer increase rate is close to zero"""
    base_classifier = BaseAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
    )
    buffered_classifier = BaseBufferedAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
        buffer_timeout_max=100,
        buffer_timeout_scheduler=get_increasing_buffer_scheduler(0.00000001),
    )

    dataset_0 = synth.STAGGER(classification_function=0, seed=0)
    dataset_1 = synth.STAGGER(classification_function=1, seed=0)
    dataset_2 = synth.STAGGER(classification_function=2, seed=0)
    t = -1
    for dataset in [dataset_0, dataset_1, dataset_2] * 3:
        for i, (x, y) in enumerate(dataset.take(500)):
            t += 1
            p_baseline = base_classifier.predict_one(x, t)
            p_buffered = buffered_classifier.predict_one(x, t)
            assert p_baseline == p_buffered

            base_classifier.learn_one(x, y, timestep=t)
            buffered_classifier.learn_one(x, y, timestep=t)


def test_representations() -> None:
    """Test predictions are the same as made by a base classifier when buffer increase rate is close to zero"""
    buffered_classifier = BaseBufferedAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
        buffer_timeout_max=100,
        buffer_timeout_scheduler=get_increasing_buffer_scheduler(1.0),
    )

    dataset_0 = synth.STAGGER(classification_function=0, seed=0)
    dataset_1 = synth.STAGGER(classification_function=1, seed=0)
    dataset_2 = synth.STAGGER(classification_function=2, seed=0)
    t = -1
    for dataset in [dataset_0, dataset_1, dataset_2] * 3:
        for i, (x, y) in enumerate(dataset.take(500)):
            t += 1
            _ = buffered_classifier.predict_one(x, t)
            buffered_classifier.learn_one(x, y, timestep=t)

            representation_window = buffered_classifier.active_window_state_representations[
                buffered_classifier.active_state_id
            ].supervised_window
            if len(representation_window) > 0:
                assert buffered_classifier.active_window_state_representations[
                    buffered_classifier.active_state_id
                ].meta_feature_values[0] == approx(np.mean([not x[1] for x in representation_window]))
