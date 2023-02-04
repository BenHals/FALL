""" A wrapper around river classifiers to provide an indicator when classification behaviour changes significantly.
"""
from typing import Optional

from river.base.typing import ClfTarget
from river.tree import HoeffdingTreeClassifier
from river.tree.splitter import Splitter


class EvolutionHoeffdingTree(HoeffdingTreeClassifier):
    def __init__(
        self,
        grace_period: int = 200,
        max_depth: Optional[int] = None,
        split_criterion: str = "info_gain",
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: Optional[list] = None,
        splitter: Splitter = None,
        binary_split: bool = False,
        max_size: float = 100,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
    ):
        super().__init__(
            grace_period,
            max_depth,
            split_criterion,
            delta,
            tau,
            leaf_prediction,
            nb_threshold,
            nominal_attributes,
            splitter,
            binary_split,
            max_size,
            memory_estimate_period,
            stop_mem_management,
            remove_poor_attrs,
            merit_preprune,
        )
        self.evolutions = 0

    def learn_one(self, x: dict, y: ClfTarget, *, sample_weight: int = 1) -> ClfTarget:
        """Wrapper around the learn_one method which increments the evolution count if the tree is updated."""
        prior_nodes = self.n_nodes
        prior_branches = self.n_branches
        prior_leaves = self.n_leaves
        p = super().learn_one(x, y, sample_weight=sample_weight)
        if self.n_nodes != prior_nodes or self.n_branches != prior_branches or self.n_leaves != prior_leaves:
            self.evolutions += 1
        return p
