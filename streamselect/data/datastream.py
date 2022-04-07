from typing import Counter, List, Optional

from streamselect.data.utils import Concept, ConceptSegment


def make_stream_concepts(
    concepts: List[Concept],
    transition_pattern: List[int],
    max_segment_length: Optional[int],
    segment_length_ratio: float = 1.0,
    boost_first_occurence: float = 1.0,
) -> List[ConceptSegment]:
    """Returns a list of concept segments describing transitions in a given dataset,
    given a list of possible concepts and a transition pattern.

    Parameters
    ----------
    concepts: List[Concept]
        A list of concept objects containing a length and underlying dataset.

    transition_pattern: List[int]
        A list of indexes into concepts describing the order of concept segments.

    max_segment_length: Optional[int]
        The maximum length a segment may be.
        Set to None for no maximum.
        (Note: a maximum must be set for concepts with an unlimited data generator,
        or when segment_length_ratio is set to -1).

    segment_length_ratio: float
        Default: 1.0
        The proportion of a concepts data.n_samples() to use for each segment, up
        to max_segment_length.
        Use -1 to force max_segment_length to be taken by repeating samples.
        When an unlimited data source is used, this parameter is ignored.

    boost_first_occurence: float
        Default: 1.0
        A multiplier in the number of observations for the first segment of each concept.
        May be set above 1.0 to encorage concepts to be stored for demonstration purposes.

    Returns
    -------
    stream_concepts: List[ConceptSegments]
        A list of non overlapping concept segments describing the data stream, where each segment
        has a start and end, and refers to one concept.
    """

    stream_concepts: List[ConceptSegment] = []
    start = -1
    end = -1
    recurrence_count: Counter[str] = Counter()
    for concept_idx in transition_pattern:
        segment_concept = concepts[concept_idx]
        if segment_concept.data.n_samples is not None and segment_length_ratio > 0:
            concept_length = int(segment_concept.data.n_samples * segment_length_ratio)
        else:
            assert (
                max_segment_length is not None
            ), "Max segment length cannot be None when concepts do not have a defined n_samples."

            concept_length = max_segment_length
        segment_length = min(concept_length, max_segment_length) if max_segment_length is not None else concept_length
        if recurrence_count[segment_concept.name] == 0:
            segment_length = int(segment_length * boost_first_occurence)

        start = end + 1
        end = start + segment_length - 1
        segment = ConceptSegment(segment_concept, start, end, recurrence_count[segment_concept.name])
        stream_concepts.append(segment)
        recurrence_count[segment_concept.name] += 1

    return stream_concepts
