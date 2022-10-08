import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from streamselect.data.utils import ConceptSegment
from streamselect.evaluation.utils import get_index_colors


def plot_ground_truth_contexts(concept_segments: list[ConceptSegment]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(15, 1.5))
    index_colors = get_index_colors()

    plot_height = 1.0
    bottom_margin = 0.5

    y = 0.0
    gt_context_segments = [[[segment.segment_start, y], [segment.segment_end, y]] for segment in concept_segments]
    gt_context_colors = [index_colors[segment.concept_idx] for segment in concept_segments]
    gt_context_lines = LineCollection(segments=gt_context_segments, colors=gt_context_colors)
    ax.add_collection(gt_context_lines)

    annotation_y = y - 0.05 * plot_height
    alpha_val = 0.8
    for segment in concept_segments:
        segment_midpoint = (segment.segment_start + segment.segment_end) / 2
        segment_color = index_colors[segment.concept_idx]
        ax.annotate(
            str(segment.concept_idx),
            xy=(segment_midpoint, annotation_y),
            xycoords="data",
            bbox={"boxstyle": "circle", "fc": "white", "ec": segment_color, "alpha": alpha_val},
            fontsize="xx-small",
            ha="center",
            va="top",
            alpha=alpha_val,
        )

        ax.axvline(x=segment.segment_end, ymin=bottom_margin - 0.1, ymax=bottom_margin, color=segment_color)

    ax.text(s="Ground Truth Context", x=0, y=y + 0.02 * plot_height, va="bottom")
    ax.set_ylim(y - (plot_height * bottom_margin), y + (plot_height * (1 - bottom_margin)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.set_xlabel("Stream Observation")
    fig.tight_layout()
    ax.plot()


if __name__ == "__main__":
    from streamselect.data.datastream import make_stream_concepts
    from streamselect.data.synthetic.wind_sim import WindSimGenerator
    from streamselect.data.transition_patterns import circular_transition_pattern
    from streamselect.data.utils import Concept

    s0 = WindSimGenerator(concept=0)
    s1 = WindSimGenerator(concept=1)
    s2 = WindSimGenerator(concept=2)
    s3 = WindSimGenerator(concept=3)
    c0 = Concept(s0, "0")
    c1 = Concept(s1, "1")
    c2 = Concept(s2, "2")
    c3 = Concept(s3, "3")

    n_repeats = 5
    pattern = circular_transition_pattern(4, n_repeats, 1.0, 1, 0.0, shuffle_order=False)

    segment_length = 1000
    concept_segments = make_stream_concepts([c0, c1, c2, c3], pattern, segment_length)

    plot_ground_truth_contexts(concept_segments)
    plt.show()
