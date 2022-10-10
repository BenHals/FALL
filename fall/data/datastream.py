from typing import Counter, Iterator, List, Optional, Tuple, Union

import numpy as np
from river.datasets.base import Dataset
from river.utils.skmultiflow_utils import check_random_state

from fall.data.utils import Concept, ConceptSegment


def make_stream_concepts(
    concepts: List[Concept],
    transition_pattern: List[int],
    max_segment_length: Optional[int],
    segment_length_ratio: float = 1.0,
    boost_first_occurence: float = 1.0,
    segment_lengths: Optional[list[int]] = None,
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
    segment_idx = 0

    if segment_lengths is not None and len(segment_lengths) == 0:
        segment_lengths = None

    for concept_idx in transition_pattern:
        segment_concept = concepts[concept_idx]
        if segment_lengths is None:
            if segment_concept.data.n_samples is not None and segment_length_ratio > 0:
                concept_length = int(segment_concept.data.n_samples * segment_length_ratio)
            else:
                assert (
                    max_segment_length is not None
                ), "Max segment length cannot be None when concepts do not have a defined n_samples."

                concept_length = max_segment_length
            segment_length = (
                min(concept_length, max_segment_length) if max_segment_length is not None else concept_length
            )
        else:
            segment_length = segment_lengths[segment_idx % len(segment_lengths)]
        if recurrence_count[segment_concept.name] == 0:
            segment_length = int(segment_length * boost_first_occurence)

        start = end + 1
        end = start + segment_length - 1
        segment = ConceptSegment(segment_concept, start, end, recurrence_count[segment_concept.name], concept_idx)
        stream_concepts.append(segment)
        recurrence_count[segment_concept.name] += 1
        segment_idx += 1

    return stream_concepts


class ConceptSegmentDataStream(Dataset):
    """Generates a stream made up of concept segments.

    Each concept represents a pure joint distribution between x and y.
    This class represents streams made up of segments each drawn from a
    particular concept. Concepts may reoccur across the stream when
    multiple segments are drawn from the same concept.

    Segments are separated by concept drift. This may be abrupt, where
    the change is distribution is instant, or gradual, where there is an
    increasing probability of being drawn from the next concept over some
    window. We use the sigmoid function to represent this.
    """

    def __init__(
        self,
        concept_segments: List[ConceptSegment],
        drifts: Union[int, List[int]],
        seed: Union[int, np.random.RandomState, None] = None,
    ) -> None:
        self.concept_segments = concept_segments
        if len(concept_segments) <= 1:
            raise AttributeError("Only a single concept passed.")

        if isinstance(drifts, int):
            self.drifts = [drifts] * (len(self.concept_segments) - 1)
        else:
            self.drifts = drifts
        if len(self.drifts) != len(concept_segments) - 1:
            raise AttributeError("Drifts size does not match number of concept segments.")

        self.seed = seed

        # Check consistancy
        first_stream = concept_segments[0].concept.data
        n_classes = first_stream.n_classes
        for seg in concept_segments[1:]:
            if seg.concept.data.n_features != first_stream.n_features:
                raise AttributeError("Inconsistent n_features.")
            if seg.concept.data.n_outputs != first_stream.n_outputs:
                raise AttributeError("Inconsistent n_features.")
            n_classes = max(n_classes, seg.concept.data.n_classes)

        super().__init__(
            n_features=first_stream.n_features,
            n_classes=n_classes,
            n_outputs=first_stream.n_outputs,
            task=first_stream.task,
        )

        self.n_samples = concept_segments[-1].segment_end - concept_segments[0].segment_start

        self.current_sample_idx = concept_segments[0].segment_start
        self.in_prev_window = False
        self.in_next_window = False
        self.seg_idx = 0

    def get_initial_concept(self) -> int:
        return self.concept_segments[0].concept_idx

    def get_current_concept(self) -> ConceptSegment:
        return self.concept_segments[self.seg_idx]

    def get_last_image(self) -> np.ndarray:
        current_segment = self.get_current_concept()
        return current_segment.get_last_image()

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        rng = check_random_state(self.seed)
        stream_iterators = {seg.concept.name: iter(seg.concept.data) for seg in self.concept_segments}

        while True:
            # If we are within the window_size of the previous drift, we have a chance of drawing from the
            # previous concept.
            current_seg = self.get_current_concept()
            observation_seg_idx = self.seg_idx
            if self.in_prev_window:
                width = self.drifts[self.seg_idx - 1]
                v = -4.0 * float(self.current_sample_idx - current_seg.segment_start) / (float(width) / 2)
                probability = 1.0 / (1.0 + np.exp(v))
                # This is the probability of the second (current) concept
                # so chance of rolling above is the probability of previous segment.
                if rng.rand() > probability:
                    observation_seg_idx -= 1

                # Don't need to check after the width.
                if self.current_sample_idx - current_seg.segment_start > width / 2:
                    self.in_prev_window = False

            elif self.in_next_window:
                width = self.drifts[self.seg_idx]
                v = -4.0 * float(self.current_sample_idx - current_seg.segment_end) / (float(width) / 2)
                probability = 1.0 / (1.0 + np.exp(v))
                # This is the probability of the second (next) concept
                # so chance of rolling below is the probability of the next segment.
                if rng.rand() <= probability:
                    observation_seg_idx += 1

            observation_seg = self.concept_segments[observation_seg_idx]
            stream = stream_iterators[observation_seg.concept.name]

            try:
                yield next(stream)
            except StopIteration:
                return

            self.current_sample_idx += 1
            if self.current_sample_idx >= self.n_samples:
                break
            next_seg = False
            if self.current_sample_idx > current_seg.segment_end:
                self.seg_idx += 1
                next_seg = True
                self.in_next_window = False

            # Handle drift windows
            width = self.drifts[self.seg_idx] if self.seg_idx < len(self.drifts) else 0
            if width > 1:
                if next_seg:
                    self.in_prev_window = True
                elif self.current_sample_idx >= current_seg.segment_end - width / 2:
                    self.in_next_window = True
