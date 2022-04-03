from river import synth
from river.drift import ADWIN
from river.tree import HoeffdingTreeClassifier

from streamselect.adaptive_learning import (
    BaseAdaptiveLearner,
    BaseBufferedAdaptiveLearner,
)
from streamselect.concept_representations import ErrorRateRepresentation
from streamselect.repository import AbsoluteValueComparer


def test_init() -> None:
    """Test initialization correctly initializes the base class."""
    buffer_timeout = 5
    al_classifier = BaseBufferedAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
        buffer_timeout=buffer_timeout,
    )

    # Check initial state has been constructed
    assert len(al_classifier.repository.states) == 1
    assert al_classifier.active_state_id in al_classifier.repository.states
    assert al_classifier.active_state_id in al_classifier.active_window_state_representations

    # Assert background state was constructed
    assert al_classifier.background_state
    assert al_classifier.background_state_active_representation
    assert al_classifier.background_state_detector
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
        window_size=window_size,
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
        buffer_timeout=0.0,
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


def test_buffer_lag() -> None:
    """Test that each state only learns from observations buffer_timeout prior to the most recent observations."""
    buffer_timeout = 50
    buffered_classifier = BaseBufferedAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        background_state_mode="drift_reset",
        buffer_timeout=buffer_timeout,
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
                or buffered_classifier.get_active_state().last_trained_active_timestep
                == t - buffer_timeout  # Test that we never train on unbuffered obs
            )

            t += 1
