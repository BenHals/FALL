import argparse
import logging
from collections import deque
from typing import List

import matplotlib
import matplotlib.animation as animation
import matplotlib.colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from river.metrics import Accuracy, CohenKappa
from river.utils import Rolling

from streamselect.adaptive_learning.base import PerformanceMonitor

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


def handle_merges_and_deletion(history, merges, deletions):
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


def segment_history(history, ex):
    diff = np.insert(history[1:] == history[:-1], 0, False)
    idx = (np.arange(history.shape[0]) - history.shape[0]) + ex
    starts = np.vstack((history[~diff], idx[~diff])).T
    return starts


def plot_TM(ax, classifier, concept_colors, c_init):
    T = classifier.concept_transitions_standard
    G = nx.DiGraph()
    for ID in classifier.state_repository:
        G.add_node(ID)
        G.add_edge(ID, ID)
        G.add_node(f"T-{ID}")
        G.add_edge(f"T-{ID}", f"T-{ID}")
    for from_id, from_T in T.items():
        total_T = from_T["total"]
        for to_id, n_T in [(i, t) for i, t in from_T.items() if i != "total"]:
            if to_id != from_id and n_T > 0:
                G.add_edge(from_id, to_id, weight=n_T, label=n_T)

    node_colors = []
    node_edges = []
    for n in G.nodes:
        try:
            ID = int(float(str(n).split("-")[-1]))
        except:
            ID = 5
        node_colors.append(concept_colors[(ID + c_init) % len(concept_colors)])
        node_edges.append(
            concept_colors[(ID + c_init) % len(concept_colors)] if ID != classifier.active_state_id else "black"
        )

    emit_edge_labels = {(n1, n2): f"{d['label']:.0f}" for n1, n2, d in G.edges(data=True) if n1 != n2}
    pos = nx.circular_layout(G)
    nx.draw(G, pos=pos, with_labels=True, ax=ax, node_color=node_colors, edgecolors=node_edges)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=emit_edge_labels, ax=ax)


class Monitor:
    def __init__(self):
        self.history_len = 500
        self.count = 0
        self.concept_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.merges = {}
        self.deletions = []
        self.print(concept_colors)
        self.acc = Accuracy()
        self.acc_baseline = Accuracy()
        self.x_history = deque(maxlen=self.history_len)
        self.acc_history = deque(maxlen=self.history_len)
        self.acc_c_history = deque(maxlen=self.history_len)
        self.baseline_history = deque(maxlen=self.history_len)
        self.likelihood_history = {}
        self.likelihood_segments = []
        self.likelihood_segment_colors = []
        self.gt_segments = deque(maxlen=self.history_len)
        self.gt_history = deque(maxlen=self.history_len)
        self.gt_colors = deque(maxlen=self.history_len)
        self.sys_segments = deque(maxlen=self.history_len)
        self.sys_history = deque(maxlen=self.history_len)
        self.sys_colors = deque(maxlen=self.history_len)
        self.sys_nomr_segments = deque(maxlen=self.history_len)
        self.sys_nomr_colors = deque(maxlen=self.history_len)
        self.recall = 0
        self.precision = 0
        self.F1 = 0
        self.adwin_likelihood_estimate = {}
        self.last_state = -1

        self.concept_cm = {}
        self.gt_totals = {}
        self.sys_totals = {}

        self.rolling_acc = Rolling(Accuracy(), window_size=100)
        self.kappa = CohenKappa()
        self.drift_count = 1000

    def plot(
        self,
        frame,
        stream,
        classifier,
        classifier_baseline,
        history_len,
        im,
        acc_line,
        acc_c_line,
        baseline_line,
        gt_lc,
        sys_lc,
        sys_nomr_lc,
        likelihood_lc,
        text_obs,
        flush=False,
    ):
        X, y = next(stream)
        if X is not None:
            p = classifier.predict_one(X)
            try:
                with np.errstate(all="ignore"):
                    p_base = classifier_baseline.predict_one(X)
            except Exception as e:
                print(f"error {e}")
                p_base = [0]

            perf_monitor: PerformanceMonitor = classifier.performance_monitor

            adwin_likelihood_estimate = {}
            # adwin_likelihood_estimate.update({i:s.get_estimated_likelihood() for i,s in classifier.state_repository.items()})
            adwin_likelihood_estimate.update(
                {perf_monitor.initial_active_state_id: perf_monitor.active_state_relevance}
            )
            self.acc.update(p[0], y[0])
            self.acc_baseline.update(p_base[0], y[0])
            self.rolling_acc.update(p[0], y[0])
            self.kappa.update(p[0], y[0])
            self.x_history.append(classifier.ex)
            self.acc_history.append(self.acc.get())
            self.baseline_history.append(self.acc_baseline.get())
            self.acc_c_history.append(self.rolling_acc.get())

            # Update concept history
            for concept_id in adwin_likelihood_estimate:
                concept_hist = self.likelihood_history.setdefault(concept_id, deque(maxlen=history_len))
                concept_hist.append((classifier.ex, adwin_likelihood_estimate[concept_id]))

            # remove any deleted concepts
            for concept_id in list(self.likelihood_history.keys()):
                if concept_id not in adwin_likelihood_estimate:
                    self.likelihood_history.pop(concept_id)

            likelihood_segments = []
            likelihood_segment_colors = []
            for concept_id, concept_hist in self.likelihood_history.items():
                seg = []
                seg_len = len(concept_hist)
                for px, py in concept_hist:
                    seg.append((px, py))
                likelihood_segments.append(seg)
                likelihood_segment_colors.append(
                    self.concept_colors[(concept_id + stream.init_concept) % len(self.concept_colors)]
                )

            curr_gt_concept = stream.concept
            curr_sys_concept = classifier.active_state_id
            self.gt_history.append(curr_gt_concept)
            self.sys_history.append(curr_sys_concept)
            if curr_gt_concept not in self.concept_cm:
                self.concept_cm[curr_gt_concept] = {}
                self.gt_totals[curr_gt_concept] = 0
            if curr_sys_concept not in self.concept_cm[curr_gt_concept]:
                self.concept_cm[curr_gt_concept][curr_sys_concept] = 0
            if curr_sys_concept not in sys_totals:
                self.sys_totals[curr_sys_concept] = 0
            self.concept_cm[curr_gt_concept][curr_sys_concept] += 1
            self.gt_totals[curr_gt_concept] += 1
            self.sys_totals[curr_sys_concept] += 1

            recall = self.concept_cm[curr_gt_concept][curr_sys_concept] / self.gt_totals[curr_gt_concept]
            precision = self.concept_cm[curr_gt_concept][curr_sys_concept] / self.sys_totals[curr_sys_concept]
            F1 = 2 * ((precision * recall) / (precision + recall))

            np_sys_history = np.array(self.sys_history)
            self.merges.update(classifier.merges if hasattr(classifier, "merges") else {})
            deletions += classifier.deletions if hasattr(classifier, "deletions") else []
            sys_h, sys_merge, sys_repair = handle_merges_and_deletion(np_sys_history, self.merges, deletions)

            gt_seg_starts = segment_history(np.array(self.gt_history), classifier.ex)
            gt_segments = []
            gt_colors = []
            seg_end = classifier.ex
            for line in gt_seg_starts[::-1]:
                gt_segments.append([[line[1], 0], [seg_end, 0]])
                gt_colors.append(self.concept_colors[line[0] % len(self.concept_colors)])
                seg_end = line[1]

            sys_seg_starts = segment_history(sys_repair, classifier.ex)
            sys_segments = []
            sys_colors = []
            seg_end = classifier.ex
            for line in sys_seg_starts[::-1]:
                sys_segments.append([[line[1], 0.75], [seg_end, 0.75]])
                sys_colors.append(self.concept_colors[(line[0] + stream.init_concept) % len(self.concept_colors)])
                seg_end = line[1]

            sys_nomr_seg_starts = segment_history(sys_h, classifier.ex)

            seg_end = classifier.ex
            for line in sys_nomr_seg_starts[::-1]:
                self.sys_nomr_segments.append([[line[1], 0.25], [seg_end, 0.25]])
                self.sys_nomr_colors.append(
                    self.concept_colors[(line[0] + stream.init_concept) % len(self.concept_colors)]
                )
                seg_end = line[1]

            classifier.partial_fit(X, y, classes=list(range(0, 21)))
            classifier_baseline.partial_fit(X, y, classes=list(range(0, 21)))
        z = stream.get_last_image()
        artists = []
        if count % 50 == 0:
            self.sample_text.set_text(f"Sample: {classifier.ex}")
            self.next_drift_text.set_text(f"Next Drift in: {1000 - (classifier.ex % 1000)}")
            self.acc_text.set_text(f"Accuracy: {self.acc.get():.2%}")
            self.r_acc_text.set_text(f"Rolling: {self.rolling_acc.get():.2%}")
            self.baseline_text.set_text(f"Baseline: {self.acc_baseline.get():.2%}")
            self.kappaM_text.set_text(f"CohenKappa: {self.kappa.get():.2%}")

            if len(self.sys_history) > 1:
                self.sys_text.set_text(f"System State: {self.sys_history[-1]}")
                self.gt_text.set_text(f"GT Concept: {self.gt_history[-1]}")
            self.recall_text.set_text(f"Recall: {recall:.2f}")
            self.precision_text.set_text(f"precision: {precision:.2f}")
            self.F1_text.set_text(f"F1: {F1:.2f}")

            self.concept_likelihood_text.set_text(
                f"Concept Likelihoods: {', '.join('{0}: {1:.2%}'.format(k, v) for k,v in adwin_likelihood_estimate.items())}"
            )
            self.merge_text.set_text(f"Merges: {' '.join('{0} -> {1}'.format(k, v) for k,v in self.merges.items())}")
            self.deletion_text.set_text(f"Deletions: {str(deletions)}")

        if last_state != classifier.active_state_id:
            self.ax8.clear()
            plot_TM(self.ax8, classifier, self.concept_colors, stream.init_concept)
            self.ax8.relim()
            self.ax8.autoscale_view(False, True, True)
            x_lim = self.ax8.get_xlim()
            x_lim_range = x_lim[1] - x_lim[0]
            self.ax8.set_xlim([x_lim[0] - x_lim_range * 0.2, x_lim[1] + x_lim_range * 0.2])
            y_lim = self.ax8.get_ylim()
            y_lim_range = y_lim[1] - y_lim[0]
            self.ax8.set_ylim([y_lim[0] - y_lim_range * 0.2, y_lim[1] + y_lim_range * 0.2])
            last_state = classifier.active_state_id
            self.fig.canvas.resize_event()

        if count % 12 == 0:
            # if count % 1 == 0:
            # ax.clear()
            # plt.clf()

            # plt.imshow(z, norm = matplotlib.colors.Normalize(0, 255))
            im.set_data(z)
        if count % 2 == 0:
            acc_line.set_data(list(self.x_history), list(self.acc_history))
            acc_c_line.set_data(list(self.x_history), list(self.acc_c_history))
            baseline_line.set_data(list(self.x_history), list(self.baseline_history))
            self.ax2.set_xlim([max(0, classifier.ex - (history_len - 1)), max(history_len, classifier.ex + 1)])
            self.ax3.set_xlim([max(0, classifier.ex - (history_len - 1)), max(history_len, classifier.ex + 1)])
            self.ax6.set_xlim([max(0, classifier.ex - (history_len - 1)), max(history_len, classifier.ex + 1)])
            gt_lc.set_segments(gt_segments)
            gt_lc.set_color(gt_colors)
            sys_lc.set_segments(sys_segments)
            sys_lc.set_color(sys_colors)
            sys_nomr_lc.set_segments(self.sys_nomr_segments)
            sys_nomr_lc.set_color(self.sys_nomr_colors)
            likelihood_lc.set_segments(likelihood_segments)
            likelihood_lc.set_color(likelihood_segment_colors)
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

        count += 1
        if count >= self.drift_count:
            self.current_concept += 1
            stream.set_concept((self.current_concept % self.n_concepts) + 1)
            count = 0
        return artists

    def run_monitor(self, stream, classifier, classifier_baseline) -> None:

        self.fig = plt.figure(figsize=(10, 5))
        gs = self.fig.add_gridspec(7, 5, height_ratios=[0.5, 0.5, 0.5, 1, 1, 1, 1], width_ratios=[1, 1, 1, 1, 1.5])
        gs.update(wspace=0.025, hspace=0.005)
        ax1 = self.fig.add_subplot(gs[0:3, :3])
        ax1.axis("off")

        self.ax2 = self.fig.add_subplot(gs[3, :4])
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

        self.ax3 = self.fig.add_subplot(gs[4, :4])
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

        self.ax6 = self.fig.add_subplot(gs[5, :4])
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

        self.ax4 = self.fig.add_subplot(gs[0:3, 3])
        self.ax4.set_frame_on(False)
        self.ax4.axis("off")

        self.ax5 = self.fig.add_subplot(gs[3:5, 4])
        self.ax5.axis("off")
        self.ax5.set_frame_on(True)
        self.ax5.set_xlim([0, 1])
        self.ax5.set_ylim([0, 1])

        self.ax7 = self.fig.add_subplot(gs[6, :4])
        self.ax7.set_xlim([0, self.history_len])
        self.ax7.axis("off")
        self.ax7.set_frame_on(True)

        self.ax8 = self.fig.add_subplot(gs[5:7, 4])
        for spine in self.ax8.spines.values():
            spine.set_visible(False)
        self.ax8.tick_params(top="off", bottom="off", left="off", right="off", labelleft="off", labelbottom="on")
        self.ax8.set_xlim([-2, 2])
        self.ax8.set_ylim([-2, 2])
        self.ax8.set_title("System FSM")

        IMAGE_SIZE = 100
        array = np.zeros(shape=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        array[0, 0] = 99  # this value allow imshow to initialise it's color scale
        im = ax1.imshow(array, cmap="gray", norm=matplotlib.colors.Normalize(0, 255))
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

        self.sample_text = self.ax4.text(
            1, 0.9, "Sample:      ", clip_on=False, transform=self.ax4.transAxes, horizontalalignment="right"
        )
        self.next_drift_text = self.ax4.text(1, 0.7, "Next Drift in:      ", horizontalalignment="right")

        self.acc_text = self.ax5.text(0.1, 0.91, "Accuracy:      ", horizontalalignment="left")
        self.ax5.plot([0.01, 0.05], [0.95, 0.95], "r-")
        self.r_acc_text = self.ax5.text(0.1, 0.82, "Rolling:      ", horizontalalignment="left")
        self.ax5.plot([0.01, 0.05], [0.85, 0.85], "g-")
        self.kappaM_text = self.ax5.text(0.1, 0.73, "CohenKappa:      ", horizontalalignment="left")
        self.ax5.plot([0.01, 0.05], [0.69, 0.69], "b-")
        self.baseline_text = self.ax5.text(0.1, 0.65, "Baseline", horizontalalignment="left")

        self.sys_text = self.ax5.text(0.1, 0.5, "System State:      ", horizontalalignment="left")
        self.gt_text = self.ax5.text(0.1, 0.4, "GT Concept:      ", horizontalalignment="left")
        self.recall_text = self.ax5.text(0.1, 0.3, "Recall:      ", horizontalalignment="left")
        self.precision_text = self.ax5.text(0.1, 0.2, "precision:      ", horizontalalignment="left")
        self.F1_text = self.ax5.text(0.1, 0.1, "F1:      ", horizontalalignment="left")
        self.concept_likelihood_text = self.ax7.text(0, 0.8, "Concept Likelihoods: ", horizontalalignment="left")
        self.merge_text = self.ax7.text(0, 0.5, "Merges: ", horizontalalignment="left")
        self.deletion_text = self.ax7.text(0, 0.2, "Deletions: ", horizontalalignment="left")
        text_obs = {
            "sample_text": self.sample_text,
            "next_drift_text": self.next_drift_text,
            "acc_text": self.acc_text,
            "r_acc_text": self.r_acc_text,
            "baseline_text": self.baseline_text,
            "kappaM_text": self.kappaM_text,
            "sys_text": self.sys_text,
            "gt_text": self.gt_text,
            "recall_text": self.recall_text,
            "precision_text": self.precision_text,
            "F1_text": self.F1_text,
            "concept_likelihood_text": self.concept_likelihood_text,
            "merge_text": self.merge_text,
            "deletion_text": self.deletion_text,
        }
        gs.tight_layout(self.fig, rect=[0, 0, 1, 1], w_pad=0.05, h_pad=0.05)
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
