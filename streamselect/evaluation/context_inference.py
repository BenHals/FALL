import statistics
from typing import Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from river.datasets.synth.stagger import STAGGER
from river.drift import ADWIN
from river.tree.hoeffding_tree_classifier import HoeffdingTreeClassifier

from streamselect.adaptive_learning.base import BaseAdaptiveLearner
from streamselect.concept_representations.error_rate_representation import (
    ErrorRateRepresentation,
)
from streamselect.data.datastream import ConceptSegmentDataStream, make_stream_concepts
from streamselect.data.transition_patterns import circular_transition_pattern
from streamselect.data.utils import Concept, ConceptSegment, StateSegment
from streamselect.evaluation.utils import (
    convert_segment_to_timeseries,
    convert_timeseries_to_segments,
    get_index_colors,
)
from streamselect.repository import AbsoluteValueComparer


def _draw_ground_truth_contexts(
    concept_segments: list[ConceptSegment],
    ax: Axes,
    plot_height: float,
    bottom_margin: float,
    y: float,
    label_loc: str = "standard",
) -> None:
    index_colors = get_index_colors()
    gt_context_segments = [[[segment.segment_start, y], [segment.segment_end, y]] for segment in concept_segments]
    gt_context_colors = [index_colors[segment.concept_idx] for segment in concept_segments]
    gt_context_lines = LineCollection(segments=gt_context_segments, colors=gt_context_colors)
    ax.add_collection(gt_context_lines)

    annotation_y_dist = 0.05
    annotation_y = y - annotation_y_dist * plot_height
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

        ax.vlines(x=segment.segment_end, ymin=y - 2 * annotation_y_dist * plot_height, ymax=y, color=segment_color)
    if label_loc == "flipped":
        ax.text(s="Ground Truth Context", x=0, y=y - (3 * annotation_y_dist + 0.02) * plot_height, va="top")
    else:
        ax.text(s="Ground Truth Context", x=0, y=y + 0.02 * plot_height, va="bottom")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.set_xlabel("Stream Observation")


def plot_ground_truth_contexts(concept_segments: list[ConceptSegment]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(15, 1.5))

    y = 0.0
    plot_height = 1.0
    bottom_margin = 0.5

    _draw_ground_truth_contexts(concept_segments, ax, plot_height, bottom_margin, y)
    ax.set_ylim(y - (plot_height * bottom_margin), y + (plot_height * (1 - bottom_margin)))

    fig.tight_layout()
    ax.plot()


def _draw_active_state_history(
    active_state_history: list[StateSegment],
    ax: Axes,
    plot_height: float,
    bottom_margin: float,
    y: float,
    label_loc: str = "standard",
) -> None:
    index_colors = get_index_colors()
    gt_context_segments = [[[segment.segment_start, y], [segment.segment_end, y]] for segment in active_state_history]
    gt_context_colors = [index_colors[segment.concept_idx] for segment in active_state_history]
    gt_context_lines = LineCollection(segments=gt_context_segments, colors=gt_context_colors)
    ax.add_collection(gt_context_lines)

    annotation_y_dist = 0.05
    annotation_y = y + annotation_y_dist * plot_height
    alpha_val = 0.8
    for segment in active_state_history:
        segment_midpoint = (segment.segment_start + segment.segment_end) / 2
        segment_color = index_colors[segment.concept_idx]
        ax.annotate(
            str(segment.concept_idx),
            xy=(segment_midpoint, annotation_y),
            xycoords="data",
            bbox={"boxstyle": "circle", "fc": "white", "ec": segment_color, "alpha": alpha_val},
            fontsize="xx-small",
            ha="center",
            va="bottom",
            alpha=alpha_val,
        )

        ax.vlines(x=segment.segment_end, ymax=y + 2 * annotation_y_dist * plot_height, ymin=y, color=segment_color)

    if label_loc == "flipped":
        ax.text(s="Active State History", x=0, y=y + (3 * annotation_y_dist + 0.02) * plot_height, va="bottom")
    else:
        ax.text(s="Active State History", x=0, y=y - 0.02 * plot_height, va="top")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_ticks([])
    ax.set_xlabel("Stream Observation")


def plot_active_state_history(active_state_history: list[StateSegment]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(15, 1.5))

    y = 0.0
    plot_height = 1.0
    bottom_margin = 0.5

    _draw_active_state_history(active_state_history, ax, plot_height, bottom_margin, y)

    ax.set_ylim(y - (plot_height * bottom_margin), y + (plot_height * (1 - bottom_margin)))
    fig.tight_layout()
    ax.plot()


def plot_system_performance(concept_segments: list[ConceptSegment], active_state_history: list[StateSegment]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(15, 3))

    y1 = 0.0
    plot_height = 1.0
    bottom_margin = 0.5

    _draw_ground_truth_contexts(concept_segments, ax, plot_height, bottom_margin, y1, label_loc="flipped")
    y2 = 0.5
    _draw_active_state_history(active_state_history, ax, plot_height, bottom_margin, y2, label_loc="flipped")

    ax.set_ylim(y1 - (plot_height * bottom_margin), y2 + (plot_height * (1 - bottom_margin)))

    fig.tight_layout()
    ax.plot()


def plot_system_context_recall(
    concept_segments: list[ConceptSegment],
    active_state_history: list[StateSegment],
    context_focus: Union[int, list[int]],
) -> None:

    fig, ax = plt.subplots(1, 1, figsize=(15, 3))

    y1 = 0.0
    plot_height = 1.0
    bottom_margin = 0.5

    _draw_ground_truth_contexts(concept_segments, ax, plot_height, bottom_margin, y1, label_loc="flipped")
    y2 = 0.5
    _draw_active_state_history(active_state_history, ax, plot_height, bottom_margin, y2, label_loc="flipped")

    context_state_recall = calculate_context_to_state_recall(concept_segments, active_state_history)
    if isinstance(context_focus, int):
        context_focus = [context_focus]

    for focus_idx in context_focus:
        assert focus_idx in context_state_recall, "Please pass a focus_idx which is in the set of contexts."
        focus_context_recall = context_state_recall[focus_idx]
        max_recall_state_context_pair_state = max(  # pylint: disable=W0640
            focus_context_recall, key=lambda k: focus_context_recall[k][2]  # pylint: disable=W0640
        )  # pylint: disable=W0640

        (
            max_recall_state_context_pair_matched_timesteps,
            max_recall_state_context_pair_unmatched_timesteps,
            _,
        ) = focus_context_recall[max_recall_state_context_pair_state]

        matched_segments = convert_timeseries_to_segments(max_recall_state_context_pair_matched_timesteps)
        unmatched_segments = convert_timeseries_to_segments(max_recall_state_context_pair_unmatched_timesteps)
        for segment in matched_segments:
            if not segment.concept_idx:
                continue

            ax.add_patch(
                mpatches.Rectangle(
                    (segment.segment_start, y1),
                    segment.segment_end - segment.segment_start,
                    y2 - y1,
                    facecolor=(0.0, 1.0, 0.0, 0.5),
                    ec="none",
                    lw=2,
                    ls="--",
                )
            )

        for segment in unmatched_segments:
            if not segment.concept_idx:
                continue

            ax.add_patch(
                mpatches.Rectangle(
                    (segment.segment_start, y1),
                    segment.segment_end - segment.segment_start,
                    y2 - y1,
                    facecolor=(1.0, 0.0, 0.0, 0.5),
                    ec="none",
                    lw=2,
                    ls="--",
                )
            )

    ax.set_ylim(y1 - (plot_height * bottom_margin), y2 + (plot_height * (1 - bottom_margin)))

    fig.tight_layout()
    ax.plot()


def plot_system_context_precision(
    concept_segments: list[ConceptSegment],
    active_state_history: list[StateSegment],
    context_focus: Union[int, list[int]],
) -> None:

    fig, ax = plt.subplots(1, 1, figsize=(15, 3))

    y1 = 0.0
    plot_height = 1.0
    bottom_margin = 0.5

    _draw_ground_truth_contexts(concept_segments, ax, plot_height, bottom_margin, y1, label_loc="flipped")
    y2 = 0.5
    _draw_active_state_history(active_state_history, ax, plot_height, bottom_margin, y2, label_loc="flipped")

    context_state_precision = calculate_context_to_state_precision(concept_segments, active_state_history)
    if isinstance(context_focus, int):
        context_focus = [context_focus]

    for focus_idx in context_focus:
        assert focus_idx in context_state_precision, "Please pass a focus_idx which is in the set of contexts."
        focus_context_precision = context_state_precision[focus_idx]  # pylint: disable=W0640
        max_precision_state_context_pair_state = max(
            focus_context_precision, key=lambda k: focus_context_precision[k][2]  # pylint: disable=W0640
        )

        (
            max_precision_state_context_pair_matched_timesteps,
            max_precision_state_context_pair_unmatched_timesteps,
            _,
        ) = focus_context_precision[max_precision_state_context_pair_state]

        matched_segments = convert_timeseries_to_segments(max_precision_state_context_pair_matched_timesteps)
        unmatched_segments = convert_timeseries_to_segments(max_precision_state_context_pair_unmatched_timesteps)
        for segment in matched_segments:
            if not segment.concept_idx:
                continue

            ax.add_patch(
                mpatches.Rectangle(
                    (segment.segment_start, y1),
                    segment.segment_end - segment.segment_start,
                    y2 - y1,
                    facecolor=(0.0, 1.0, 0.0, 0.5),
                    ec="none",
                    lw=2,
                    ls="--",
                )
            )

        for segment in unmatched_segments:
            if not segment.concept_idx:
                continue

            ax.add_patch(
                mpatches.Rectangle(
                    (segment.segment_start, y1),
                    segment.segment_end - segment.segment_start,
                    y2 - y1,
                    facecolor=(1.0, 0.0, 0.0, 0.5),
                    ec="none",
                    lw=2,
                    ls="--",
                )
            )

    ax.set_ylim(y1 - (plot_height * bottom_margin), y2 + (plot_height * (1 - bottom_margin)))

    fig.tight_layout()
    ax.plot()


def calculate_context_to_state_recall(
    concept_segments: list[ConceptSegment], active_state_history: list[StateSegment]
) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray, float]]]:
    concept_timeseries = convert_segment_to_timeseries(concept_segments)
    active_state_timeseries = convert_segment_to_timeseries(active_state_history)

    concept_indexes = np.unique(concept_timeseries)
    active_state_indexes = np.unique(active_state_timeseries)

    all_state_context_recall_values: dict[int, dict[int, tuple[np.ndarray, np.ndarray, float]]] = {}
    for concept_idx in concept_indexes:
        concept_mask = concept_timeseries == concept_idx

        concept_state_recall_values: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        for state_idx in active_state_indexes:
            state_mask = active_state_timeseries == state_idx

            context_and_state_mask = concept_mask & state_mask
            context_and_not_state_mask = concept_mask & (~state_mask)

            concept_state_recall_values[state_idx] = (
                context_and_state_mask,
                context_and_not_state_mask,
                context_and_state_mask.sum() / concept_mask.sum(),
            )

        all_state_context_recall_values[concept_idx] = concept_state_recall_values

    return all_state_context_recall_values


def calculate_context_to_state_precision(
    concept_segments: list[ConceptSegment], active_state_history: list[StateSegment]
) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray, float]]]:
    concept_timeseries = convert_segment_to_timeseries(concept_segments)
    active_state_timeseries = convert_segment_to_timeseries(active_state_history)

    concept_indexes = np.unique(concept_timeseries)
    active_state_indexes = np.unique(active_state_timeseries)

    all_state_context_precision_values: dict[int, dict[int, tuple[np.ndarray, np.ndarray, float]]] = {}
    for concept_idx in concept_indexes:
        concept_mask = concept_timeseries == concept_idx

        concept_state_precision_values: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
        for state_idx in active_state_indexes:
            state_mask = active_state_timeseries == state_idx

            state_and_context_mask = concept_mask & state_mask
            state_and_not_context_mask = (~concept_mask) & state_mask

            concept_state_precision_values[state_idx] = (
                state_and_context_mask,
                state_and_not_context_mask,
                state_and_context_mask.sum() / state_mask.sum(),
            )

        all_state_context_precision_values[concept_idx] = concept_state_precision_values

    return all_state_context_precision_values


def calculate_context_to_state_max_f1(
    concept_segments: list[ConceptSegment], active_state_history: list[StateSegment]
) -> dict[int, float]:
    context_state_precision = calculate_context_to_state_precision(concept_segments, active_state_history)
    context_state_recall = calculate_context_to_state_recall(concept_segments, active_state_history)

    concept_indexes = {segment.concept_idx for segment in concept_segments}
    active_state_indexes = {segment.concept_idx for segment in active_state_history}

    context_to_state_max_f1_scores: dict[int, float] = {}
    for concept_idx in concept_indexes:
        state_f1_scores: list[tuple[float, int]] = []
        for state_idx in active_state_indexes:
            _, _, precision = context_state_precision[concept_idx][state_idx]
            _, _, recall = context_state_recall[concept_idx][state_idx]
            f1 = 0.0
            if recall + precision > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            print(precision, recall, f1)
            state_f1_scores.append((f1, state_idx))

        max_f1, _ = max(state_f1_scores)
        context_to_state_max_f1_scores[concept_idx] = max_f1

    return context_to_state_max_f1_scores


def calculate_CF1_score(concept_segments: list[ConceptSegment], active_state_history: list[StateSegment]) -> float:
    context_to_state_max_f1_scores = calculate_context_to_state_max_f1(concept_segments, active_state_history)
    print(context_to_state_max_f1_scores)
    return statistics.mean(list(context_to_state_max_f1_scores.values()))


if __name__ == "__main__":
    s0 = STAGGER(classification_function=0)
    s1 = STAGGER(classification_function=1)
    s2 = STAGGER(classification_function=2)
    c0 = Concept(s0, "0")
    c1 = Concept(s1, "1")
    c2 = Concept(s2, "2")

    n_repeats = 5
    pattern = circular_transition_pattern(3, n_repeats, 1.0, 1, 0.0, shuffle_order=False)

    segment_length = 1000
    concept_segments = make_stream_concepts([c0, c1, c2], pattern, segment_length)

    seed = 42
    datastream = ConceptSegmentDataStream(concept_segments, 0, seed)

    classifier = BaseAdaptiveLearner(
        classifier_constructor=HoeffdingTreeClassifier,
        representation_constructor=ErrorRateRepresentation,
        representation_comparer=AbsoluteValueComparer(),
        drift_detector_constructor=ADWIN,
        representation_window_size=50,
        representation_update_period=5,
        drift_detection_mode="lower",
    )

    for i, (X, y) in enumerate(datastream.take(datastream.n_samples)):
        classifier.predict_one(X, i)
        classifier.learn_one(X, y, timestep=i)

    # plot_ground_truth_contexts(concept_segments)
    # plot_active_state_history(classifier.performance_monitor.active_state_history)
    # plot_system_performance(concept_segments, classifier.performance_monitor.active_state_history)
    # calculate_context_to_state_recall(concept_segments, classifier.performance_monitor.active_state_history)
    # plot_system_context_recall(concept_segments, classifier.performance_monitor.active_state_history, [0, 1])
    plot_system_context_precision(concept_segments, classifier.performance_monitor.active_state_history, [0, 1])
    print(calculate_CF1_score(concept_segments, classifier.performance_monitor.active_state_history))
    plt.show()
