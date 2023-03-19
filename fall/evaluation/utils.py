from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from fall.data.utils import DataStreamSegment

RGBAColor = tuple[float, float, float, float]


def get_index_colors() -> list[RGBAColor]:
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def convert_segment_to_timeseries(segments: Sequence[DataStreamSegment]) -> np.ndarray:
    segment_timeseries = [
        np.full(segment.segment_end - segment.segment_start + 1, segment.concept_idx) for segment in segments
    ]
    return np.concatenate(segment_timeseries)


def convert_timeseries_to_segments(timeseries: np.ndarray) -> list[DataStreamSegment]:
    initial_segment = DataStreamSegment(0, 0, timeseries[0])
    segments = [initial_segment]

    is_new_segment = timeseries[:-1] != timeseries[1:]
    for i, is_segment_start in enumerate(is_new_segment):
        if not is_segment_start:
            continue
        last_segment_end = i
        next_segment_start = i + 1
        segments[-1].segment_end = last_segment_end
        next_segment = DataStreamSegment(next_segment_start, next_segment_start, timeseries[next_segment_start])
        segments.append(next_segment)
    segments[-1].segment_end = timeseries.shape[0]
    return segments
