from collections import deque
from typing import Counter, Deque, Iterator, List, Tuple

from pytest import approx
from river import synth
from river.datasets import Bananas
from river.datasets.base import BINARY_CLF, SyntheticDataset

from streamselect.data.datastream import ConceptSegmentDataStream, make_stream_concepts
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


def test_concept_segment_datastream() -> None:
    """Testing the ConceptSegmentDatastream class.
    Test abrupt predictions are equal.
    """
    seed = 42
    s0 = synth.STAGGER(classification_function=0, seed=seed)
    s1 = synth.STAGGER(classification_function=1, seed=seed + 1)
    s2 = synth.STAGGER(classification_function=2, seed=seed + 2)

    c0 = Concept(s0, "0")
    c1 = Concept(s1, "1")
    c2 = Concept(s2, "2")

    concept_segments = [
        ConceptSegment(c0, 0, 2, 0),
        ConceptSegment(c1, 3, 5, 0),
        ConceptSegment(c2, 6, 8, 0),
        ConceptSegment(c0, 9, 11, 0),
        ConceptSegment(c1, 12, 14, 0),
        ConceptSegment(c2, 15, 17, 0),
    ]

    datastream = ConceptSegmentDataStream(concept_segments, 0, seed)

    d0 = iter(s0)
    d1 = iter(s1)
    d2 = iter(s2)
    test_against = [next(d) for d in [d0, d1, d2, d0, d1, d2] for i in range(3)]

    for t, (x, y) in enumerate(datastream):
        assert x == test_against[t][0]
        assert y == test_against[t][1]


def test_concept_segment_datastream_2() -> None:
    """Testing the ConceptSegmentDatastream class.
    Test abrupt predictions are equal using the constructor helpers.
    """
    seed = 42
    s0 = synth.STAGGER(classification_function=0, seed=seed)
    s1 = synth.STAGGER(classification_function=1, seed=seed + 1)
    s2 = synth.STAGGER(classification_function=2, seed=seed + 2)
    c0 = Concept(s0, "0")
    c1 = Concept(s1, "1")
    c2 = Concept(s2, "2")

    n_repeats = 30
    pattern = circular_transition_pattern(3, n_repeats, 1.0, 1, 0.0, shuffle_order=False)

    segment_length = 400
    concept_segments = make_stream_concepts([c0, c1, c2], pattern, segment_length)

    datastream = ConceptSegmentDataStream(concept_segments, 0, seed)

    # Testing against
    d0 = iter(s0)
    d1 = iter(s1)
    d2 = iter(s2)
    test_against = [next(d) for d in [d0, d1, d2] * n_repeats for i in range(segment_length)]

    for t, (x, y) in enumerate(datastream):
        assert x == test_against[t][0]
        assert y == test_against[t][1]


class MockStream(SyntheticDataset):
    def __init__(self, v: int):
        super().__init__(task=BINARY_CLF, n_features=1, n_classes=3, n_outputs=1)
        self.v = v
        self.current_sample_idx = 0

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        while True:
            yield self.v, self.v


def test_concept_segment_datastream_3() -> None:
    """Testing the ConceptSegmentDatastream class.
    Test abrupt predictions with mock class.
    """
    seed = 42
    s0 = MockStream(v=0)
    s1 = MockStream(v=1)
    s2 = MockStream(v=2)

    c0 = Concept(s0, "0")
    c1 = Concept(s1, "1")
    c2 = Concept(s2, "2")

    n_repeats = 3
    pattern = circular_transition_pattern(3, n_repeats, 1.0, 1, 0.0, shuffle_order=False)

    segment_length = 10000
    concept_segments = make_stream_concepts([c0, c1, c2], pattern, segment_length)

    datastream = ConceptSegmentDataStream(concept_segments, 0, seed)

    # Testing against
    recent_concepts: Deque[int] = deque()
    recent_concept_counts: Counter[int] = Counter()
    assertations: List[Tuple[int, int, float]] = [
        # t, concept, proportion of recent
        (100, 0, 1.0),
        (1000, 0, 1.0),
        (5000, 0, 1.0),
        (9999, 0, 1.0),
        (15000, 1, 1.0),
        (19999, 1, 1.0),
        (25000, 2, 1.0),
        (29999, 2, 1.0),
        (35000, 0, 1.0),
        (39999, 0, 1.0),
        (45000, 1, 1.0),
        (49999, 1, 1.0),
        (55000, 2, 1.0),
        (59999, 2, 1.0),
    ]
    for t, (_, y) in enumerate(datastream):
        concept = y
        recent_concept_counts[concept] += 1
        recent_concepts.append(concept)
        if len(recent_concepts) > 100:
            old_concept = recent_concepts.popleft()
            recent_concept_counts[old_concept] -= 1

        if len(assertations) > 0 and t == assertations[0][0]:
            a = assertations.pop(0)
            # Assert that the recent proportion is approx what we expect
            assert recent_concept_counts[a[1]] / 100 == approx(a[2])


def test_concept_segment_datastream_4() -> None:
    """Testing the ConceptSegmentDatastream class.
    Test gradual predictions with mock class.
    """
    seed = 42
    s0 = MockStream(v=0)
    s1 = MockStream(v=1)
    s2 = MockStream(v=2)

    c0 = Concept(s0, "0")
    c1 = Concept(s1, "1")
    c2 = Concept(s2, "2")

    n_repeats = 3
    pattern = circular_transition_pattern(3, n_repeats, 1.0, 1, 0.0, shuffle_order=False)

    segment_length = 10000
    concept_segments = make_stream_concepts([c0, c1, c2], pattern, segment_length)

    drift_width = 3000
    datastream = ConceptSegmentDataStream(concept_segments, drift_width, seed)

    # Testing against
    recent_concepts: Deque[int] = deque()
    recent_concept_counts: Counter[int] = Counter()
    assertations: List[Tuple[int, int, float]] = [
        # t, concept, proportion of recent
        (200, 0, 1.0),
        (1000, 0, 1.0),
        (5000, 0, 1.0),
        (9999, 0, 0.5),
        (9999, 1, 0.5),
        (15000, 1, 1.0),
        (19999, 1, 0.5),
        (19999, 2, 0.5),
        (25000, 2, 1.0),
        (29999, 2, 0.5),
        (29999, 0, 0.5),
        (35000, 0, 1.0),
        (39999, 0, 0.5),
        (45000, 1, 1.0),
        (49999, 1, 0.5),
        (55000, 2, 1.0),
        (59999, 2, 0.5),
    ]
    for t, (_, y) in enumerate(datastream):
        concept = y
        recent_concept_counts[concept] += 1
        recent_concepts.append(concept)
        if len(recent_concepts) > 200:
            old_concept = recent_concepts.popleft()
            recent_concept_counts[old_concept] -= 1

        while len(assertations) > 0 and t == assertations[0][0]:
            a = assertations.pop(0)
            # Assert that the recent proportion is approx what we expect
            print(a)
            print(recent_concept_counts)
            assert recent_concept_counts[a[1]] / 200 == approx(a[2], rel=3e-1)


# %%
