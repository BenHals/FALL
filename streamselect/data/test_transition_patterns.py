from collections import Counter
from typing import Dict

import numpy as np
from pytest import approx

from streamselect.data.transition_patterns import circular_transition_pattern


def test_circular_inorder() -> None:
    """Test the circular transition pattern."""
    pattern = circular_transition_pattern(3, 3, forward_proportion=1.0, n_forward=1, noise=0.0, shuffle_order=False)
    assert pattern == [0, 1, 2, 0, 1, 2, 0, 1, 2]


def test_circular_inorder_shuffle() -> None:
    """Test the circular transition pattern."""
    seed = 42
    rng = np.random.default_rng(seed)
    order = list(range(3))
    rng.shuffle(order)
    pattern = circular_transition_pattern(
        3, 3, forward_proportion=1.0, n_forward=1, noise=0.0, shuffle_order=True, seed=seed
    )
    # Test that we shuffle correctly with the same seed
    assert pattern == [order[0], order[1], order[2], order[0], order[1], order[2], order[0], order[1], order[2]]


test_circular_inorder_shuffle()

# %%


def test_circular_transition_probabilities_falloff_1() -> None:
    """Test the circular transition pattern."""
    seed = 42
    n_concepts = 10
    n_repeats = 1000
    pattern = circular_transition_pattern(
        n_concepts, n_repeats, forward_proportion=0.5, n_forward=2, noise=0.0, shuffle_order=False, seed=seed
    )
    counts: Dict[int, Counter] = {}
    for i in range(n_concepts):
        counts[i] = Counter()

    prev_concept = pattern[0]
    for next_concept in pattern[1:]:
        counts[prev_concept][next_concept] += 1
        prev_concept = next_concept

    for i, prev_concept in enumerate(range(n_concepts)):
        next_idx = (i + 1) % n_concepts
        next_next_idx = (i + 2) % n_concepts
        assert counts[prev_concept][next_next_idx] == approx(counts[prev_concept][next_idx] * 0.5, rel=2e-1)


def test_circular_transition_probabilities_falloff_2() -> None:
    """Test the circular transition pattern."""
    seed = 42
    n_concepts = 10
    n_repeats = 1000
    pattern = circular_transition_pattern(
        n_concepts, n_repeats, forward_proportion=0.5, n_forward=3, noise=0.0, shuffle_order=False, seed=seed
    )
    counts: Dict[int, Counter] = {}
    for i in range(n_concepts):
        counts[i] = Counter()

    prev_concept = pattern[0]
    for next_concept in pattern[1:]:
        counts[prev_concept][next_concept] += 1
        prev_concept = next_concept

    for i, prev_concept in enumerate(range(n_concepts)):
        next_idx = (i + 1) % n_concepts
        next_next_idx = (i + 2) % n_concepts
        next_next_next_idx = (i + 3) % n_concepts
        assert counts[prev_concept][next_next_idx] == approx(counts[prev_concept][next_idx] * 0.5, rel=2e-1)
        assert counts[prev_concept][next_next_next_idx] == approx(counts[prev_concept][next_next_idx] * 0.5, rel=2e-1)


def test_circular_transition_probabilities_falloff_3() -> None:
    """Test the circular transition pattern."""
    seed = 42
    n_concepts = 10
    n_repeats = 1000
    pattern = circular_transition_pattern(
        n_concepts, n_repeats, forward_proportion=0.3, n_forward=3, noise=0.0, shuffle_order=False, seed=seed
    )
    counts: Dict[int, Counter] = {}
    for i in range(n_concepts):
        counts[i] = Counter()

    prev_concept = pattern[0]
    for next_concept in pattern[1:]:
        counts[prev_concept][next_concept] += 1
        prev_concept = next_concept

    for i, prev_concept in enumerate(range(n_concepts)):
        next_idx = (i + 1) % n_concepts
        next_next_idx = (i + 2) % n_concepts
        next_next_next_idx = (i + 3) % n_concepts
        assert counts[prev_concept][next_next_idx] == approx(counts[prev_concept][next_idx] * (1 - 0.3), rel=2e-1)
        assert counts[prev_concept][next_next_next_idx] == approx(
            counts[prev_concept][next_next_idx] * (1 - 0.3), rel=2e-1
        )


def test_circular_transition_probabilities_noise() -> None:
    """Test the circular transition pattern."""
    seed = 42
    n_concepts = 10
    n_repeats = 1000
    noise = 0.5
    pattern = circular_transition_pattern(
        n_concepts, n_repeats, forward_proportion=1.0, n_forward=1, noise=noise, shuffle_order=False, seed=seed
    )
    counts: Dict[int, Counter] = {}
    for i in range(n_concepts):
        counts[i] = Counter()

    prev_concept = pattern[0]
    for next_concept in pattern[1:]:
        counts[prev_concept][next_concept] += 1
        prev_concept = next_concept

    for i, prev_concept in enumerate(range(n_concepts)):
        for ni in range(n_concepts):
            if i == ni:
                continue
            if (i + 1) % n_concepts == ni:
                continue
            # assert that noise is distributed across all concepts
            assert counts[prev_concept][ni] == approx(n_repeats * (noise / (n_concepts - 1)), rel=5e-1, abs=2)
