from typing import Counter, List

from river import synth
from river.datasets import Bananas

from streamselect.data.datastream import make_stream_concepts
from streamselect.data.transition_patterns import circular_transition_pattern
from streamselect.data.utils import Concept, ConceptSegment

# from collections import Counter


def test_make_stream_concepts_synthetic() -> None:
    """Testing the stream segments made by make_stream_concepts."""
    s0 = synth.STAGGER(classification_function=0)
    s1 = synth.STAGGER(classification_function=1)
    s2 = synth.STAGGER(classification_function=2)

    c0 = Concept(data=s0, name="c0")
    c1 = Concept(data=s1, name="c1")
    c2 = Concept(data=s2, name="c2")

    concepts = [c2, c0, c1]
    pattern = circular_transition_pattern(
        len(concepts), n_repeats=3, forward_proportion=1.0, n_forward=1, noise=0.0, shuffle_order=False
    )
    stream_segments: List[ConceptSegment] = make_stream_concepts(concepts, pattern, 5000)
    # We should have a stream segment for each concept for each repeat
    assert len(stream_segments) == 3 * 3

    # Check that all indexes in the dataset are accounted for, and we only increase
    # over the dataset.
    sum_indexes = 0
    max_so_far = -1
    for c_seg in stream_segments:
        # segment_end is inclusive
        for i in range(c_seg.segment_start, c_seg.segment_end + 1):
            sum_indexes += i
            assert i > max_so_far
            max_so_far = i
    assert sum_indexes == sum(range(0, 3 * 3 * 5000))

    # check recurrences are in the right order for their respective
    # concepts
    reccurence_count: Counter[str] = Counter()
    for c_seg in stream_segments:
        assert c_seg.recurrence_count == reccurence_count[c_seg.concept.name]
        reccurence_count[c_seg.concept.name] += 1

    # Check concept order is the same as the transition pattern.
    for i, seg in enumerate(stream_segments):
        assert seg.concept.name == concepts[pattern[i]].name


def test_make_stream_concepts_synthetic_2() -> None:
    """Testing the stream segments made by make_stream_concepts.
    Test with randomized transition pattern"""
    n_repeats = 30
    s0 = synth.STAGGER(classification_function=0)
    s1 = synth.STAGGER(classification_function=1)
    s2 = synth.STAGGER(classification_function=2)

    c0 = Concept(data=s0, name="c0")
    c1 = Concept(data=s1, name="c1")
    c2 = Concept(data=s2, name="c2")

    concepts = [c2, c0, c1]
    pattern = circular_transition_pattern(
        len(concepts), n_repeats=n_repeats, forward_proportion=0.3, n_forward=3, noise=0.5, shuffle_order=True
    )
    stream_segments: List[ConceptSegment] = make_stream_concepts(concepts, pattern, 5000)
    # We should have a stream segment for each concept for each repeat
    assert len(stream_segments) == 3 * n_repeats

    # Check that all indexes in the dataset are accounted for, and we only increase
    # over the dataset.
    sum_indexes = 0
    max_so_far = -1
    for c_seg in stream_segments:
        # segment_end is inclusive
        for i in range(c_seg.segment_start, c_seg.segment_end + 1):
            sum_indexes += i
            assert i > max_so_far
            max_so_far = i
    assert sum_indexes == sum(range(0, 3 * n_repeats * 5000))

    # check recurrences are in the right order for their respective
    # concepts
    reccurence_count: Counter[str] = Counter()
    for c_seg in stream_segments:
        assert c_seg.recurrence_count == reccurence_count[c_seg.concept.name]
        reccurence_count[c_seg.concept.name] += 1

    # Check concept order is the same as the transition pattern.
    for i, seg in enumerate(stream_segments):
        assert seg.concept.name == concepts[pattern[i]].name


def test_make_stream_concepts_real() -> None:
    """Testing the stream segments made by make_stream_concepts.
    Test with randomized transition pattern"""
    n_repeats = 3
    s0 = Bananas()
    s1 = Bananas()
    s2 = Bananas()

    c0 = Concept(data=s0, name="c0")
    c1 = Concept(data=s1, name="c1")
    c2 = Concept(data=s2, name="c2")

    concepts = [c2, c0, c1]
    pattern = circular_transition_pattern(
        len(concepts), n_repeats=n_repeats, forward_proportion=1.0, n_forward=1, noise=0.0, shuffle_order=False
    )
    stream_segments: List[ConceptSegment] = make_stream_concepts(concepts, pattern, 10000)
    # We should have a stream segment for each concept for each repeat
    assert len(stream_segments) == 3 * n_repeats

    # Check that all indexes in the dataset are accounted for, and we only increase
    # over the dataset.
    sum_indexes = 0
    max_so_far = -1
    for c_seg in stream_segments:
        # segment_end is inclusive
        for i in range(c_seg.segment_start, c_seg.segment_end + 1):
            sum_indexes += i
            assert i > max_so_far
            max_so_far = i

        # Concepts should be the real length of the dataset, which is 5300, since this is less than 10000
    assert sum_indexes == sum(
        range(0, (n_repeats * s0.n_samples) + (n_repeats * s1.n_samples) + (n_repeats * s1.n_samples))  # type: ignore
    )

    # check recurrences are in the right order for their respective
    # concepts
    reccurence_count: Counter[str] = Counter()
    for c_seg in stream_segments:
        assert c_seg.recurrence_count == reccurence_count[c_seg.concept.name]
        reccurence_count[c_seg.concept.name] += 1

    # Check concept order is the same as the transition pattern.
    for i, seg in enumerate(stream_segments):
        assert seg.concept.name == concepts[pattern[i]].name


def test_make_stream_concepts_real_2() -> None:
    """Testing the stream segments made by make_stream_concepts.
    Test with randomized transition pattern"""
    n_repeats = 3
    s0 = Bananas()
    s1 = Bananas()
    s2 = Bananas()

    c0 = Concept(data=s0, name="c0")
    c1 = Concept(data=s1, name="c1")
    c2 = Concept(data=s2, name="c2")

    concepts = [c2, c0, c1]
    pattern = circular_transition_pattern(
        len(concepts), n_repeats=n_repeats, forward_proportion=1.0, n_forward=1, noise=0.0, shuffle_order=True
    )
    stream_segments: List[ConceptSegment] = make_stream_concepts(concepts, pattern, 5000)
    # We should have a stream segment for each concept for each repeat
    assert len(stream_segments) == 3 * n_repeats

    # Check that all indexes in the dataset are accounted for, and we only increase
    # over the dataset.
    sum_indexes = 0
    max_so_far = -1
    for c_seg in stream_segments:
        # segment_end is inclusive
        for i in range(c_seg.segment_start, c_seg.segment_end + 1):
            sum_indexes += i
            assert i > max_so_far
            max_so_far = i

    assert sum_indexes == sum(
        # Since the max_segment_length of 5000 is less than n_samples, this should have priority
        range(0, (n_repeats * 5000) + (n_repeats * 5000) + (n_repeats * 5000))  # type: ignore
    )

    # check recurrences are in the right order for their respective
    # concepts
    reccurence_count: Counter[str] = Counter()
    for c_seg in stream_segments:
        assert c_seg.recurrence_count == reccurence_count[c_seg.concept.name]
        reccurence_count[c_seg.concept.name] += 1

    # Check concept order is the same as the transition pattern.
    for i, seg in enumerate(stream_segments):
        assert seg.concept.name == concepts[pattern[i]].name


def test_make_stream_concepts_real_3() -> None:
    """Testing the stream segments made by make_stream_concepts.
    Test with randomized transition pattern"""
    n_repeats = 3
    s0 = Bananas()
    s1 = Bananas()
    s2 = Bananas()

    c0 = Concept(data=s0, name="c0")
    c1 = Concept(data=s1, name="c1")
    c2 = Concept(data=s2, name="c2")

    concepts = [c2, c0, c1]
    pattern = circular_transition_pattern(
        len(concepts), n_repeats=n_repeats, forward_proportion=1.0, n_forward=1, noise=0.0, shuffle_order=True
    )
    stream_segments: List[ConceptSegment] = make_stream_concepts(concepts, pattern, 10000, segment_length_ratio=-1)
    # We should have a stream segment for each concept for each repeat
    assert len(stream_segments) == 3 * n_repeats

    # Check that all indexes in the dataset are accounted for, and we only increase
    # over the dataset.
    sum_indexes = 0
    max_so_far = -1
    for c_seg in stream_segments:
        # segment_end is inclusive
        for i in range(c_seg.segment_start, c_seg.segment_end + 1):
            sum_indexes += i
            assert i > max_so_far
            max_so_far = i

    assert sum_indexes == sum(
        # Since the max_segment_length of 10000 is greater than n_samples, and segment_length_ratio = -1,
        # we take 10000 each segment.
        range(0, (n_repeats * 10000) + (n_repeats * 10000) + (n_repeats * 10000))  # type: ignore
    )

    # check recurrences are in the right order for their respective
    # concepts
    reccurence_count: Counter[str] = Counter()
    for c_seg in stream_segments:
        assert c_seg.recurrence_count == reccurence_count[c_seg.concept.name]
        reccurence_count[c_seg.concept.name] += 1

    # Check concept order is the same as the transition pattern.
    for i, seg in enumerate(stream_segments):
        assert seg.concept.name == concepts[pattern[i]].name


def test_make_stream_concepts_real_4() -> None:
    """Testing the stream segments made by make_stream_concepts.
    Test with randomized transition pattern"""
    n_repeats = 3
    s0 = Bananas()
    s1 = Bananas()
    s2 = Bananas()

    c0 = Concept(data=s0, name="c0")
    c1 = Concept(data=s1, name="c1")
    c2 = Concept(data=s2, name="c2")

    concepts = [c2, c0, c1]
    pattern = circular_transition_pattern(
        len(concepts), n_repeats=n_repeats, forward_proportion=1.0, n_forward=1, noise=0.0, shuffle_order=True
    )
    stream_segments: List[ConceptSegment] = make_stream_concepts(concepts, pattern, 10000, segment_length_ratio=0.3)
    # We should have a stream segment for each concept for each repeat
    assert len(stream_segments) == 3 * n_repeats

    # Check that all indexes in the dataset are accounted for, and we only increase
    # over the dataset.
    sum_indexes = 0
    max_so_far = -1
    for c_seg in stream_segments:
        # segment_end is inclusive
        for i in range(c_seg.segment_start, c_seg.segment_end + 1):
            sum_indexes += i
            assert i > max_so_far
            max_so_far = i

    assert sum_indexes == sum(
        # Since the max_segment_length of 10000 is greater than n_samples, and segment_length_ratio = 0.3,
        # we take 0.3 * 5200 each segment.
        range(
            0, (n_repeats * int(0.3 * 5300)) + (n_repeats * int(0.3 * 5300)) + (n_repeats * int(0.3 * 5300))
        )  # type: ignore
    )

    # check recurrences are in the right order for their respective
    # concepts
    reccurence_count: Counter[str] = Counter()
    for c_seg in stream_segments:
        assert c_seg.recurrence_count == reccurence_count[c_seg.concept.name]
        reccurence_count[c_seg.concept.name] += 1

    # Check concept order is the same as the transition pattern.
    for i, seg in enumerate(stream_segments):
        assert seg.concept.name == concepts[pattern[i]].name
