import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from streamselect.data.utils import ConceptSegment
from streamselect.evaluation.utils import get_index_colors, RGBAColor

def plot_ground_truth_contexts(concept_segments: list[ConceptSegment]) -> None:
    fig, ax = plt.subplots(1, 1)
    index_colors = get_index_colors()

    y = 0
    gt_context_segments = [[[segment.segment_start, y], [segment.segment_end, y]] for segment in concept_segments]
    gt_context_colors = [index_colors[segment.concept_idx] for segment in concept_segments]
    gt_context_lines = LineCollection(segments=gt_context_segments, colors=gt_context_colors)

    ax.add_collection(gt_context_lines)
    ax.plot()



