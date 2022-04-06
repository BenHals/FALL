""" Functions for generating concept transition patterns for synthetic data streams"""

from typing import List, Optional

import numpy as np


def circular_transition_pattern(
    n_concepts: int,
    n_repeats: int,
    forward_proportion: float,
    n_forward: int,
    noise: float,
    seed: Optional[int] = None,
    shuffle_order: bool = True,
) -> List[int]:
    """Create a circular transition pattern where concepts are shuffled into a given order, then transitions around
    this order are generated. Transitions may skip elements, and random transitions are allowed based on parameters.
    A circular pattern ensures that all concepts can appear in the future, i.e., no sinks may appear during the pattern.
    Parameters
    ----------

    n_concepts: int
        Number of concepts to include in the pattern.

    n_repeats: int
        Number of times to repeat each concept, i.e.,
        how many recurrences of each concept are present.

    forward_proportion: float
        The relative probability for each possible forward transition.
        I.E., the next concept in the order is most likely, then the next
        concept after that is proportionally forward_proportion of the remaining prob.

    n_forward: int
        How many steps around the order may be taken in one transition.

    noise: float
        The proportional chance for a random transition, disregarding the order.
        Note: probabilities are standardized to 1, so the final chance of noise
        may be different.

    seed: Optional[int]
        Default: None
        Seed for numpy rng. If none, is randomized.

    shuffle_order: bool
        Default: True
        If true, concept order is shuffled. If false, the underlying order is
        0 -> n_concepts.

    Returns
    -------
    :List[int]
        Returns a list of concept ids specifying the generated order of concepts.

    """
    if n_concepts <= 1:
        return [0] * n_repeats

    if seed is None:
        seed = np.random.randint(0, 100000)

    # Randomize order of concepts
    rng = np.random.default_rng(seed)
    concept_indexs = list(range(n_concepts))
    if shuffle_order:
        rng.shuffle(concept_indexs)

    # Calculate transition probabilities from each concept
    transition_probabilities = {}
    for idx, concept_idx in enumerate(concept_indexs):
        t_probs = {}
        # init noise probability for all concepts except current one
        noise_per_concept = noise / (n_concepts - 1)
        for next_idx in concept_indexs:
            if concept_idx == next_idx:
                continue
            t_probs[next_idx] = noise_per_concept

        p = 1.0 - noise
        for i in range(1, n_forward + 1):
            next_idx = (idx + i) % n_concepts
            next_concept_idx = concept_indexs[next_idx]
            if next_idx == idx:
                break
            state_prob = p * forward_proportion
            p = p - state_prob
            t_probs[next_concept_idx] = state_prob

        total_prob = sum(t_probs.values())
        for k in t_probs:
            t_probs[k] /= total_prob
        transition_probabilities[concept_idx] = t_probs

    # We assume each concept appears the same number of times.
    total_occurences = n_concepts * n_repeats
    current_concept = concept_indexs[0]
    transition_pattern = [current_concept]
    while len(transition_pattern) < total_occurences:
        n_idx, probs = list(zip(*transition_probabilities[current_concept].items()))
        current_concept = rng.choice(n_idx, p=probs)
        transition_pattern.append(current_concept)

    return transition_pattern
