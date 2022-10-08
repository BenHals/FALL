import argparse
import logging
from collections import deque
from typing import List, Any, Union

import matplotlib
from matplotlib.image import AxesImage
import matplotlib.animation as animation
import matplotlib.colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from river.metrics import Accuracy, CohenKappa
from river.utils import Rolling
from river.base import Classifier

from streamselect.adaptive_learning.base import PerformanceMonitor, BaseAdaptiveLearner
from streamselect.data.datastream import ConceptSegmentDataStream

Vector = List[float]

logging.basicConfig(filename="demo.log", level=logging.INFO)


def pandas_fill(arr: np.ndarray) -> np.ndarray:
    df = pd.Series(arr)
    df.fillna(method="bfill", inplace=True)
    out = df.to_numpy()
    return out


def numpy_fill(arr: np.ndarray) -> np.ndarray:
    """Solution provided by Divakar."""
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]) - 20, 0)
    np.minimum.accumulate(idx[::-1], axis=0, out=idx[::-1])
    idx = idx + 20
    out = arr[idx]
    return out


def handle_merges_and_deletion(history: np.ndarray, merges: dict[int, int], deletions: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    merge_history = np.copy(history)
    for m_init in merges:
        m_to = merges[m_init]
        while m_to in merges:
            m_from = m_to
            m_to = merges[m_from]
        merge_history = np.where(merge_history == m_init, m_to, merge_history)
    repair_history = np.copy(merge_history)
    for dm in deletions:
        repair_history = np.where(repair_history == dm, np.nan, repair_history)
    try:
        repair_history = pandas_fill(repair_history)
    except Exception as e:
        print(repair_history)
        raise e

    return history, merge_history.astype(int), repair_history.astype(int)


def segment_history(history: np.ndarray, ex: int) -> np.ndarray:
    diff = np.insert(history[1:] == history[:-1], 0, False)
    idx = (np.arange(history.shape[0]) - history.shape[0]) + ex
    starts = np.vstack((history[~diff], idx[~diff])).T
    return starts


def plot_TM(ax: plt.axis, active_state_id: int, T: dict[str, dict[str, int]], repository: dict[int, Any], concept_colors: list[str], c_init: int) -> None:
    G = nx.DiGraph()
    for ID in repository:
        G.add_node(str(ID))
        G.add_edge(str(ID), str(ID))
        G.add_node(f"T-{ID}")
        G.add_edge(f"T-{ID}", f"T-{ID}")
    for from_id, from_T in T.items():
        total_T = from_T["total"]
        for to_id, n_T in [(i, t) for i, t in from_T.items() if i != "total"]:
            if to_id != from_id and n_T > 0:
                G.add_edge(str(from_id), str(to_id), weight=n_T, label=n_T)

    node_colors = []
    node_edges = []
    for n in G.nodes:
        try:
            ID = int(float(str(n).split("-")[-1]))
        except:
            ID = 5
        node_colors.append(concept_colors[(ID + c_init) % len(concept_colors)])
        node_edges.append(
            concept_colors[(ID + c_init) % len(concept_colors)] if ID != active_state_id else "black"
        )

    emit_edge_labels = {(n1, n2): f"{d['label']:.0f}" for n1, n2, d in G.edges(data=True) if n1 != n2}
    pos = nx.circular_layout(G)
    nx.draw(G, pos=pos, with_labels=True, ax=ax, node_color=node_colors, edgecolors=node_edges)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=emit_edge_labels, ax=ax)


class Monitor:
    def __init__(self) -> None:
        self.history_len = 500
        self.count = 0
        self.ex = -1
        self.concept_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.merges: dict[int, int] = {}
        self.deletions: list[int] = []
        self.acc = Accuracy()
        self.acc_baseline = Accuracy()
        self.x_history: deque[int] = deque(maxlen=self.history_len)
        self.acc_history: deque[float] = deque(maxlen=self.history_len)
        self.acc_c_history: deque[float] = deque(maxlen=self.history_len)
        self.baseline_history: deque[float] = deque(maxlen=self.history_len)
        self.likelihood_history: dict[int, deque[tuple[int, float]]] = {}
        self.likelihood_segments: deque[list[list[Union[int, float]]]] = deque(maxlen=self.history_len)
        self.likelihood_segment_colors: list[Any] = []
        self.gt_segments: deque[list[list[Union[int, float]]]] = deque(maxlen=self.history_len)
        self.gt_history: deque[Any] = deque(maxlen=self.history_len)
        self.gt_colors: deque[Any] = deque(maxlen=self.history_len)
        self.sys_segments: deque[list[list[Union[int, float]]]] = deque(maxlen=self.history_len)
        self.sys_history: deque[Any] = deque(maxlen=self.history_len)
        self.sys_colors: deque[Any] = deque(maxlen=self.history_len)
        self.sys_nomr_segments: deque[Any] = deque(maxlen=self.history_len)
        self.sys_nomr_colors: deque[Any] = deque(maxlen=self.history_len)
        self.recall = 0.0
        self.precision = 0.0
        self.F1 = 0.0
        self.adwin_likelihood_estimate: dict[int, float] = {}
        self.last_state = -1

        self.concept_cm: dict[int, dict[int, int]] = {}
        self.gt_totals: dict[int, int] = {}
        self.sys_totals: dict[int, int] = {}

        self.rolling_acc = Rolling(Accuracy(), window_size=100)
        self.kappa = CohenKappa()
        self.drift_count = 1000
        self.current_concept = 0

        self.T: dict[str, dict[str, int]] = {}

    def plot(
        self,
        frame: Any,
        stream_iter: ConceptSegmentDataStream,
        classifier: BaseAdaptiveLearner,
        classifier_baseline: Classifier,
        history_len: int,
        im: AxesImage,
        acc_line: Any,
        acc_c_line: Any,
        baseline_line: Any,
        gt_lc: Any,
        sys_lc: Any,
        sys_nomr_lc: Any,
        likelihood_lc: Any,
        text_obs: Any,
        flush:bool=False,
    ) -> list[Any]:
        X, y = next(stream_iter)
        if X is not None:
            self.ex += 1
            p = classifier.predict_one(X)
            if p is None:
                p = 0
            print(X, y, p)
            try:
                with np.errstate(all="ignore"):
                    p_base = classifier_baseline.predict_one(X)
            except Exception as e:
                print(f"error {e}")
                p_base = [0]

            perf_monitor: PerformanceMonitor = classifier.performance_monitor

            self.adwin_likelihood_estimate = {}
            # adwin_likelihood_estimate.update({i:s.get_estimated_likelihood() for i,s in classifier.state_repository.items()})
            self.adwin_likelihood_estimate.update(
                # {perf_monitor.initial_active_state_id: perf_monitor.active_state_relevance}
                {name:relevance for name, relevance in perf_monitor.state_relevances.items()}
            )
            self.acc.update(p, y)
            self.acc_baseline.update(p_base, y)
            self.rolling_acc.update(p, y)
            self.kappa.update(p, y)
            self.x_history.append(self.ex)
            self.acc_history.append(self.acc.get())
            self.baseline_history.append(self.acc_baseline.get())
            self.acc_c_history.append(self.rolling_acc.get())

            # Update concept history
            for concept_id in self.adwin_likelihood_estimate:
                concept_hist = self.likelihood_history.setdefault(concept_id, deque(maxlen=history_len))
                concept_hist.append((self.ex, self.adwin_likelihood_estimate[concept_id]))

            # remove any deleted concepts
            for concept_id in list(self.likelihood_history.keys()):
                if concept_id not in self.adwin_likelihood_estimate:
                    self.likelihood_history.pop(concept_id)

            self.likelihood_segments = deque(maxlen=self.history_len)
            self.likelihood_segment_colors = []
            for concept_id, concept_hist in self.likelihood_history.items():
                seg = []
                seg_len = len(concept_hist)
                for px, py in concept_hist:
                    seg.append([px, py])
                self.likelihood_segments.append(seg)
                self.likelihood_segment_colors.append(
                    self.concept_colors[(concept_id) % len(self.concept_colors)]
                )

            curr_gt_concept = self.stream.concept_segments[self.stream.seg_idx].concept_idx
            curr_sys_concept = perf_monitor.initial_active_state_id
            self.gt_history.append(curr_gt_concept)
            self.sys_history.append(curr_sys_concept)
            if curr_gt_concept not in self.concept_cm:
                self.concept_cm[curr_gt_concept] = {}
                self.gt_totals[curr_gt_concept] = 0
            if curr_sys_concept not in self.concept_cm[curr_gt_concept]:
                self.concept_cm[curr_gt_concept][curr_sys_concept] = 0
            if curr_sys_concept not in self.sys_totals:
                self.sys_totals[curr_sys_concept] = 0
            self.concept_cm[curr_gt_concept][curr_sys_concept] += 1
            self.gt_totals[curr_gt_concept] += 1
            self.sys_totals[curr_sys_concept] += 1

            self.recall = self.concept_cm[curr_gt_concept][curr_sys_concept] / self.gt_totals[curr_gt_concept]
            self.precision = self.concept_cm[curr_gt_concept][curr_sys_concept] / self.sys_totals[curr_sys_concept]
            self.F1 = 2.0 * ((self.precision * self.recall) / (self.precision + self.recall))

            np_sys_history = np.array(self.sys_history)
            self.merges.update(perf_monitor.merges if hasattr(classifier, "merges") else {})
            self.deletions += perf_monitor.deletions if hasattr(classifier, "deletions") else []
            sys_h, sys_merge, sys_repair = handle_merges_and_deletion(np_sys_history, self.merges, self.deletions)

            self.gt_seg_starts = segment_history(np.array(self.gt_history), self.ex)
            self.gt_segments = deque(maxlen=self.history_len)
            self.gt_colors = deque(maxlen=self.history_len)
            seg_end = self.ex
            for line in self.gt_seg_starts[::-1]:
                self.gt_segments.append([[line[1], 0], [seg_end, 0]])
                self.gt_colors.append(self.concept_colors[line[0] % len(self.concept_colors)])
                seg_end = line[1]

            self.sys_seg_starts = segment_history(sys_repair, self.ex)
            self.sys_segments = deque(maxlen=self.history_len)
            self.sys_colors = deque(maxlen=self.history_len)
            seg_end = self.ex
            for line in self.sys_seg_starts[::-1]:
                self.sys_segments.append([[line[1], 0.75], [seg_end, 0.75]])
                self.sys_colors.append(self.concept_colors[(line[0]) % len(self.concept_colors)])
                seg_end = line[1]

            self.sys_nomr_seg_starts = segment_history(sys_h, self.ex)

            seg_end = self.ex
            for line in self.sys_nomr_seg_starts[::-1]:
                self.sys_nomr_segments.append([[line[1], 0.25], [seg_end, 0.25]])
                self.sys_nomr_colors.append(
                    self.concept_colors[(line[0]) % len(self.concept_colors)]
                )
                seg_end = line[1]

            classifier.learn_one(X, y)
            classifier_baseline.learn_one(X, y)

        z = self.stream.get_last_image()
        artists = []
        sample_text = text_obs["sample_text"]
        next_drift_text = text_obs["next_drift_text"]
        acc_text = text_obs["acc_text"]
        r_acc_text = text_obs["r_acc_text"]
        baseline_text = text_obs["baseline_text"]
        kappaM_text = text_obs["kappaM_text"]
        sys_text = text_obs["sys_text"]
        gt_text = text_obs["gt_text"]
        recall_text = text_obs["recall_text"]
        precision_text = text_obs["precision_text"]
        F1_text = text_obs["F1_text"]
        concept_likelihood_text = text_obs["concept_likelihood_text"]
        merge_text = text_obs["merge_text"]
        deletion_text = text_obs["deletion_text"]
        if self.count % 50 == 0:
            sample_text.set_text(f"Sample: {self.ex}")
            next_drift_text.set_text(f"Next Drift in: {1000 - (self.ex % 1000)}")
            acc_text.set_text(f"Accuracy: {self.acc.get():.2%}")
            r_acc_text.set_text(f"Rolling: {self.rolling_acc.get():.2%}")
            baseline_text.set_text(f"Baseline: {self.acc_baseline.get():.2%}")
            kappaM_text.set_text(f"CohenKappa: {self.kappa.get():.2%}")

            if len(self.sys_history) > 1:
                sys_text.set_text(f"System State: {self.sys_history[-1]}")
                gt_text.set_text(f"GT Concept: {self.gt_history[-1]}")
            recall_text.set_text(f"Recall: {self.recall:.2f}")
            precision_text.set_text(f"precision: {self.precision:.2f}")
            F1_text.set_text(f"F1: {self.F1:.2f}")

            concept_likelihood_text.set_text(
                f"Concept Likelihoods: {', '.join('{0}: {1:.2%}'.format(k, v) for k,v in self.adwin_likelihood_estimate.items())}"
            )
            merge_text.set_text(f"Merges: {' '.join('{0} -> {1}'.format(k, v) for k,v in self.merges.items())}")
            deletion_text.set_text(f"Deletions: {str(self.deletions)}")

        if self.last_state != classifier.active_state_id:
            self.ax8.clear()
            plot_TM(self.ax8, perf_monitor.initial_active_state_id, perf_monitor.transition_matrix, perf_monitor.repository, self.concept_colors, self.stream.get_initial_concept())
            self.ax8.relim()
            self.ax8.autoscale_view(False, True, True)
            x_lim = self.ax8.get_xlim()
            x_lim_range = x_lim[1] - x_lim[0]
            self.ax8.set_xlim([x_lim[0] - x_lim_range * 0.2, x_lim[1] + x_lim_range * 0.2])
            y_lim = self.ax8.get_ylim()
            y_lim_range = y_lim[1] - y_lim[0]
            self.ax8.set_ylim([y_lim[0] - y_lim_range * 0.2, y_lim[1] + y_lim_range * 0.2])
            self.last_state = classifier.active_state_id
            self.fig.canvas.resize_event()

        if self.count % 12 == 0:
            # if count % 1 == 0:
            # ax.clear()
            # plt.clf()

            # plt.imshow(z, norm = matplotlib.colors.Normalize(0, 255))
            im.set_data(z)
        if self.count % 2 == 0:
            acc_line.set_data(list(self.x_history), list(self.acc_history))
            acc_c_line.set_data(list(self.x_history), list(self.acc_c_history))
            baseline_line.set_data(list(self.x_history), list(self.baseline_history))
            self.ax2.set_xlim([max(0, self.ex - (history_len - 1)), max(history_len, self.ex + 1)])
            self.ax3.set_xlim([max(0, self.ex - (history_len - 1)), max(history_len, self.ex + 1)])
            self.ax6.set_xlim([max(0, self.ex - (history_len - 1)), max(history_len, self.ex + 1)])
            gt_lc.set_segments(self.gt_segments)
            gt_lc.set_color(self.gt_colors)
            sys_lc.set_segments(self.sys_segments)
            sys_lc.set_color(self.sys_colors)
            sys_nomr_lc.set_segments(self.sys_nomr_segments)
            sys_nomr_lc.set_color(self.sys_nomr_colors)
            likelihood_lc.set_segments(self.likelihood_segments)
            likelihood_lc.set_color(self.likelihood_segment_colors)
        artists = [
            im,
            acc_line,
            acc_c_line,
            baseline_line,
            gt_lc,
            sys_lc,
            sys_nomr_lc,
            likelihood_lc,
            *text_obs.values(),
        ]

        self.count += 1
        if self.count >= self.drift_count:
            self.current_concept += 1
            self.count = 0
        return artists

    def run_monitor(self, stream: ConceptSegmentDataStream, classifier: BaseAdaptiveLearner, classifier_baseline: Classifier) -> None:

        self.fig = plt.figure(figsize=(10, 5))
        self.gs = self.fig.add_gridspec(7, 5, height_ratios=[0.5, 0.5, 0.5, 1, 1, 1, 1], width_ratios=[1, 1, 1, 1, 1.5])
        self.gs.update(wspace=0.025, hspace=0.005)
        self.ax1 = self.fig.add_subplot(self.gs[0:3, :3])
        self.ax1.axis("off")

        self.ax2 = self.fig.add_subplot(self.gs[3, :4])
        self.ax2.set_xlim([0, self.history_len])
        self.ax2.set_ylim([0, 1])
        self.ax2.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )
        self.ax2.set_title("Performance")

        self.ax3 = self.fig.add_subplot(self.gs[4, :4])
        self.ax3.set_xlim([0, self.history_len])
        self.ax3.set_ylim([-0.1, 1])
        self.ax3.set_yticks([0.0, 0.25, 0.5, 0.75])
        self.ax3.set_yticklabels(
            ["Ground Truth Concepts", "System State", "Sys State w Merging", "Sys State w Repair"]
        )
        self.ax3.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )
        self.ax3.set_title("Active Concept")

        self.ax6 = self.fig.add_subplot(self.gs[5, :4])
        self.ax6.set_xlim([0, self.history_len])
        self.ax6.set_ylim([0, 1])
        self.ax6.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )
        self.ax6.set_title("Concept Likelihoods")

        self.ax4 = self.fig.add_subplot(self.gs[0:3, 3])
        self.ax4.set_frame_on(False)
        self.ax4.axis("off")

        self.ax5 = self.fig.add_subplot(self.gs[3:5, 4])
        self.ax5.axis("off")
        self.ax5.set_frame_on(True)
        self.ax5.set_xlim([0, 1])
        self.ax5.set_ylim([0, 1])

        self.ax7 = self.fig.add_subplot(self.gs[6, :4])
        self.ax7.set_xlim([0, self.history_len])
        self.ax7.axis("off")
        self.ax7.set_frame_on(True)

        self.ax8 = self.fig.add_subplot(self.gs[5:7, 4])
        for spine in self.ax8.spines.values():
            spine.set_visible(False)
        self.ax8.tick_params(top="off", bottom="off", left="off", right="off", labelleft="off", labelbottom="on")
        self.ax8.set_xlim([-2, 2])
        self.ax8.set_ylim([-2, 2])
        self.ax8.set_title("System FSM")

        IMAGE_SIZE = 100
        array = np.zeros(shape=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        array[0, 0] = 99  # this value allow imshow to initialise it's color scale
        im = self.ax1.imshow(array, cmap="gray", norm=matplotlib.colors.Normalize(0, 255))
        (acc_line,) = self.ax2.plot([], [], "r-")
        (acc_c_line,) = self.ax2.plot([], [], "g-")
        (baseline_line,) = self.ax2.plot([], [], "b-")
        gt_lc = LineCollection(self.gt_segments)
        gt_lc.set_linewidth(2)
        self.ax3.add_collection(gt_lc)
        sys_lc = LineCollection(self.sys_segments)
        sys_lc.set_linewidth(2)
        self.ax3.add_collection(sys_lc)
        sys_nomr_lc = LineCollection(self.sys_segments)
        sys_nomr_lc.set_linewidth(2)
        self.ax3.add_collection(sys_nomr_lc)

        likelihood_lc = LineCollection(self.gt_segments)
        likelihood_lc.set_linewidth(2)
        self.ax6.add_collection(likelihood_lc)

        sample_text = self.ax4.text(
            1, 0.9, "Sample:      ", clip_on=False, transform=self.ax4.transAxes, horizontalalignment="right"
        )
        next_drift_text = self.ax4.text(1, 0.7, "Next Drift in:      ", horizontalalignment="right")

        acc_text = self.ax5.text(0.1, 0.91, "Accuracy:      ", horizontalalignment="left")
        self.ax5.plot([0.01, 0.05], [0.95, 0.95], "r-")
        r_acc_text = self.ax5.text(0.1, 0.82, "Rolling:      ", horizontalalignment="left")
        self.ax5.plot([0.01, 0.05], [0.85, 0.85], "g-")
        kappaM_text = self.ax5.text(0.1, 0.73, "CohenKappa:      ", horizontalalignment="left")
        self.ax5.plot([0.01, 0.05], [0.69, 0.69], "b-")
        baseline_text = self.ax5.text(0.1, 0.65, "Baseline", horizontalalignment="left")

        sys_text = self.ax5.text(0.1, 0.5, "System State:      ", horizontalalignment="left")
        gt_text = self.ax5.text(0.1, 0.4, "GT Concept:      ", horizontalalignment="left")
        recall_text = self.ax5.text(0.1, 0.3, "Recall:      ", horizontalalignment="left")
        precision_text = self.ax5.text(0.1, 0.2, "precision:      ", horizontalalignment="left")
        F1_text = self.ax5.text(0.1, 0.1, "F1:      ", horizontalalignment="left")
        concept_likelihood_text = self.ax7.text(0, 0.8, "Concept Likelihoods: ", horizontalalignment="left")
        merge_text = self.ax7.text(0, 0.5, "Merges: ", horizontalalignment="left")
        deletion_text = self.ax7.text(0, 0.2, "Deletions: ", horizontalalignment="left")
        text_obs = {
            "sample_text": sample_text,
            "next_drift_text": next_drift_text,
            "acc_text": acc_text,
            "r_acc_text": r_acc_text,
            "baseline_text": baseline_text,
            "kappaM_text": kappaM_text,
            "sys_text": sys_text,
            "gt_text": gt_text,
            "recall_text": recall_text,
            "precision_text": precision_text,
            "F1_text": F1_text,
            "concept_likelihood_text": concept_likelihood_text,
            "merge_text": merge_text,
            "deletion_text": deletion_text,
        }
        self.gs.tight_layout(self.fig, rect=[0, 0, 1, 1], w_pad=0.05, h_pad=0.05)
        self.stream = stream
        stream_iter = iter(stream)
        ani = animation.FuncAnimation(
            self.fig,
            self.plot,
            fargs=(
                stream_iter,
                classifier,
                classifier_baseline,
                self.history_len,
                im,
                acc_line,
                acc_c_line,
                baseline_line,
                gt_lc,
                sys_lc,
                sys_nomr_lc,
                likelihood_lc,
                text_obs,
                False,
            ),
            interval=0.001,
            blit=True,
        )
        plt.show()
