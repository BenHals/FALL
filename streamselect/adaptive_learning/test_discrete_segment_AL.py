from river import synth
from river.drift import ADWIN
from river.tree import HoeffdingTreeClassifier

from streamselect.adaptive_learning import (
    BaseAdaptiveLearner,
    BaseBufferedAdaptiveLearner,
    BufferedDiscreteSegmentAL,
    DiscreteSegmentAL,
    get_constant_max_buffer_scheduler,
)
from streamselect.concept_representations import ErrorRateRepresentation
from streamselect.repository import AbsoluteValueComparer


def test_base_predictions() -> None:
    """Test predictions are the same as made by a base classifier before a transition"""
    base_classifier = BaseAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
    )
    discrete_classifier = DiscreteSegmentAL(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
    )

    dataset_0 = synth.STAGGER(classification_function=0, seed=0)
    for dataset in [dataset_0] * 1:
        for t, (x, y) in enumerate(dataset.take(500)):
            p_baseline = base_classifier.predict_one(x, t)
            p_discrete = discrete_classifier.predict_one(x, t)
            assert p_baseline == p_discrete

            base_classifier.learn_one(x, y, timestep=t)
            discrete_classifier.learn_one(x, y, timestep=t)


def test_discrete_states() -> None:
    """Test that each segment is handled by a discrete state, and only one state is kept in memory."""
    discrete_classifier = DiscreteSegmentAL(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
    )

    dataset_0 = synth.STAGGER(classification_function=0, seed=0)
    dataset_1 = synth.STAGGER(classification_function=1, seed=0)
    dataset_2 = synth.STAGGER(classification_function=2, seed=0)
    detected_segments = 1
    states = {discrete_classifier.active_state_id}
    t = 0
    for dataset in [dataset_0, dataset_1, dataset_2] * 3:
        for x, y in dataset.take(500):
            _ = discrete_classifier.predict_one(x, t)
            discrete_classifier.learn_one(x, y, timestep=t)
            states.add(discrete_classifier.active_state_id)
            t += 1

            # The classifier should only have one state at a time, the active one
            assert len(discrete_classifier.repository.states) == 1

            # We should observe a unique active_state_id for each concept drift detected
            if discrete_classifier.performance_monitor.in_drift:
                detected_segments += 1
            assert len(states) == detected_segments


def test_buffered_base_predictions() -> None:
    """Test predictions are the same as made by a base classifier before a transition"""
    buffer_timeout = 25
    base_classifier = BaseBufferedAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
        buffer_timeout_max=buffer_timeout,
        buffer_timeout_scheduler=get_constant_max_buffer_scheduler(),
    )
    discrete_classifier = BufferedDiscreteSegmentAL(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        buffer_timeout_max=buffer_timeout,
        buffer_timeout_scheduler=get_constant_max_buffer_scheduler(),
    )

    dataset_0 = synth.STAGGER(classification_function=0, seed=0)
    for dataset in [dataset_0] * 1:
        for t, (x, y) in enumerate(dataset.take(500)):
            p_baseline = base_classifier.predict_one(x, t)
            p_discrete = discrete_classifier.predict_one(x, t)
            if base_classifier.performance_monitor.last_drift or discrete_classifier.performance_monitor.last_drift:
                print(base_classifier.performance_monitor.last_drift)
                break
            assert p_baseline == p_discrete

            base_classifier.learn_one(x, y, timestep=t)
            discrete_classifier.learn_one(x, y, timestep=t)


def test_buffered_discrete_states() -> None:
    """Test that each segment is handled by a discrete state, and only one state is kept in memory."""
    buffer_timeout = 25
    discrete_classifier = BufferedDiscreteSegmentAL(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        buffer_timeout_max=buffer_timeout,
        buffer_timeout_scheduler=get_constant_max_buffer_scheduler(),
    )

    dataset_0 = synth.STAGGER(classification_function=0, seed=0)
    dataset_1 = synth.STAGGER(classification_function=1, seed=0)
    dataset_2 = synth.STAGGER(classification_function=2, seed=0)
    detected_segments = 1
    states = {discrete_classifier.active_state_id}
    t = 0
    for dataset in [dataset_0, dataset_1, dataset_2] * 3:
        for x, y in dataset.take(500):
            _ = discrete_classifier.predict_one(x, t)
            discrete_classifier.learn_one(x, y, timestep=t)
            states.add(discrete_classifier.active_state_id)
            t += 1

            # The classifier should only have one state at a time, the active one
            assert len(discrete_classifier.repository.states) == 1

            # We should observe a unique active_state_id for each concept drift detected
            if discrete_classifier.performance_monitor.in_drift:
                detected_segments += 1
            assert len(states) == detected_segments
