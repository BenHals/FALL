import numpy as np
from river.naive_bayes import GaussianNB

from streamselect.adaptive_learning.classifier_adaptation import (
    max_acc_sig_relevance_adaptation,
    maximum_relevance_adaptation,
)
from streamselect.concept_representations import ErrorRateRepresentation
from streamselect.repository import Repository
from streamselect.states import State


def test_maximum_relevance() -> None:
    """This strategy should return the state from R u B with maximum relevance."""
    bg = State(GaussianNB(), lambda x: ErrorRateRepresentation(1, x))
    repo = Repository(
        classifier_constructor=GaussianNB, representation_constructor=lambda x: ErrorRateRepresentation(1, x)
    )
    s1 = repo.add_next_state()
    s2 = repo.add_next_state()

    s1_relevance = np.random.normal(0.95, 0.025, 1000)
    for rel in s1_relevance:
        s1.add_active_state_relevance(rel)
    s1_accuracy = np.random.normal(0.95, 0.025, 1000)
    for acc in s1_accuracy:
        s1.in_concept_accuracy_record.update(acc)

    s2_relevance = np.random.normal(0.925, 0.025, 1000)
    for rel in s2_relevance:
        s2.add_active_state_relevance(rel)
    s2_accuracy = np.random.normal(0.975, 0.025, 1000)
    for acc in s2_accuracy:
        s2.in_concept_accuracy_record.update(acc)

    bg_relevance = np.random.normal(0.85, 0.025, 1000)
    for rel in bg_relevance:
        bg.add_active_state_relevance(rel)
    bg_accuracy = np.random.normal(0.99, 0.025, 1000)
    for acc in bg_accuracy:
        bg.in_concept_accuracy_record.update(acc)

    state_relevance = {bg.state_id: 0.85, s1.state_id: 0.95, s2.state_id: 0.925}

    # We should select s1 because it has the highest relevance
    assert maximum_relevance_adaptation(bg, repo, state_relevance, None) == s1


def test_max_acc_sig_relevance() -> None:
    """This strategy should return the state from R u B with maximum accuracy from those with maximum relevance."""
    bg = State(GaussianNB(), lambda x: ErrorRateRepresentation(1, x))
    repo = Repository(
        classifier_constructor=GaussianNB, representation_constructor=lambda x: ErrorRateRepresentation(1, x)
    )
    s1 = repo.add_next_state()
    s2 = repo.add_next_state()

    s1_relevance = np.random.normal(0.95, 0.025, 1000)
    for rel in s1_relevance:
        s1.add_active_state_relevance(rel)
    s1_accuracy = np.random.normal(0.95, 0.025, 1000)
    for acc in s1_accuracy:
        s1.in_concept_accuracy_record.update(acc)

    s2_relevance = np.random.normal(0.925, 0.025, 1000)
    for rel in s2_relevance:
        s2.add_active_state_relevance(rel)
    s2_accuracy = np.random.normal(0.975, 0.025, 1000)
    for acc in s2_accuracy:
        s2.in_concept_accuracy_record.update(acc)

    bg_relevance = np.random.normal(0.85, 0.025, 1000)
    for rel in bg_relevance:
        bg.add_active_state_relevance(rel)
    bg_accuracy = np.random.normal(0.99, 0.025, 1000)
    for acc in bg_accuracy:
        bg.in_concept_accuracy_record.update(acc)

    state_relevance = {bg.state_id: 0.85, s1.state_id: 0.95, s2.state_id: 0.925}

    # We should select s2 because its relevance is not significantly different to s1
    # and it has higher accuracy.
    assert max_acc_sig_relevance_adaptation(bg, repo, state_relevance, None) == s2
