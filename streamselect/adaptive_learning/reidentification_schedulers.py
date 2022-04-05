""" Provides classes for scheduling concept re-identification beyond standard drift detection.
For example, we may want to recheck re-identification some length of time after a drift. """
import abc
import heapq
from enum import Enum
from functools import total_ordering
from typing import Dict, List, Optional


class DriftType(Enum):
    DriftDetectorTriggered = 0
    ScheduledOne = 1
    ScheduledPeriodic = 2


@total_ordering
class DriftInfo:
    """Holds information about a drift"""

    def __init__(self, timestep: int, drift_type: DriftType = DriftType.DriftDetectorTriggered) -> None:
        self.drift_timestep = timestep
        self.drift_type = drift_type
        self.triggered_transition: Optional[bool] = None
        self.transitioned_from: Optional[int] = None
        self.transitioned_to: Optional[int] = None
        self.reidentification_relevance: Optional[Dict[int, float]] = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DriftInfo):
            return NotImplemented
        return self.drift_timestep == other.drift_timestep

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, DriftInfo):
            return NotImplemented
        return self.drift_timestep < other.drift_timestep

    def __str__(self) -> str:
        if not self.triggered_transition:
            return f"D({self.drift_type}@{self.drift_timestep})"
        return (
            f"D({self.drift_type}@{self.drift_timestep}"
            + f"-{self.reidentification_relevance}|{self.transitioned_from}->{self.transitioned_to})"
        )

    def __repr__(self) -> str:
        return str(self)


class BaseReidentificationScheduler(abc.ABC):
    """Base scheduler, when called on a drift returns a set of reidentification checks
    scheduled in the future."""

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def get_scheduled_checks(self, drift: DriftInfo) -> List[DriftInfo]:
        """Returns a list of future drifts to schedule from the given drift."""

    @abc.abstractmethod
    def transition_reset(self, transition_timestep: int) -> List[DriftInfo]:
        """Returns a list of future drifts to schedule after a transition."""

    @abc.abstractmethod
    def get_initialization_checks(self, initial_timestep: int) -> List[DriftInfo]:
        """Returns a list of future drifts to schedule initially."""


class DriftDetectionCheck(BaseReidentificationScheduler):
    def __init__(self, check_delay: int):
        """Schedules a reidentification check check_delay after any drift detector triggered drift."""
        super().__init__()
        self.check_delay = check_delay

    def get_scheduled_checks(self, drift: DriftInfo) -> List[DriftInfo]:
        scheduled_checks = []
        if drift.drift_type == DriftType.DriftDetectorTriggered:
            scheduled_checks.append(DriftInfo(drift.drift_timestep + self.check_delay, DriftType.ScheduledOne))
        return scheduled_checks

    def transition_reset(self, transition_timestep: int) -> List[DriftInfo]:
        scheduled_checks = []
        scheduled_checks.append(DriftInfo(transition_timestep + self.check_delay, DriftType.ScheduledOne))
        return scheduled_checks

    def get_initialization_checks(self, initial_timestep: int) -> List[DriftInfo]:
        return []


class PeriodicCheck(BaseReidentificationScheduler):
    def __init__(self, check_period: int):
        """Schedules a reidentification check every check_period steps, reset after a transition."""
        super().__init__()
        self.check_period = check_period

    def get_scheduled_checks(self, drift: DriftInfo) -> List[DriftInfo]:
        scheduled_checks = []
        if drift.drift_type == DriftType.ScheduledPeriodic:
            scheduled_checks.append(DriftInfo(drift.drift_timestep + self.check_period, DriftType.ScheduledPeriodic))
        return scheduled_checks

    def transition_reset(self, transition_timestep: int) -> List[DriftInfo]:
        scheduled_checks = []
        scheduled_checks.append(DriftInfo(transition_timestep + self.check_period, DriftType.ScheduledPeriodic))
        return scheduled_checks

    def get_initialization_checks(self, initial_timestep: int) -> List[DriftInfo]:
        scheduled_checks = []
        scheduled_checks.append(DriftInfo(initial_timestep + self.check_period, DriftType.ScheduledPeriodic))
        return scheduled_checks


class ReidentificationSchedule:
    """Keeps track of scheduled drifts using a set of schedulers."""

    def __init__(self) -> None:
        self.schedulers: List[BaseReidentificationScheduler] = []
        self.scheduled_drifts: List[DriftInfo] = []

    def add_scheduler(self, scheduler: BaseReidentificationScheduler) -> None:
        """Add a new scheduler."""
        self.schedulers.append(scheduler)

    def schedule_reidentification(self, drift: DriftInfo) -> None:
        """Takes a current drift and adds any new scheduled drifts which arise based on the set of
        added schedulers. Note that we don't schedule any reidentifications before or at the timestep of
        the passed drift.

        Parameters
        ----------
        drift: DriftInfo
            Information describing the current drift.
        """
        for scheduler in self.schedulers:
            new_checks = scheduler.get_scheduled_checks(drift)
            for check in new_checks:
                if check.drift_timestep <= drift.drift_timestep:
                    continue
                heapq.heappush(self.scheduled_drifts, check)

    def get_scheduled_reidentifications(self, current_timestep: int) -> List[DriftInfo]:
        """Returns and re-identification checks scheduled for or prior to the current timestep."""
        scheduled_reidentification: List[DriftInfo] = []
        while len(self.scheduled_drifts) > 0 and self.scheduled_drifts[0].drift_timestep <= current_timestep:
            scheduled_reidentification.append(heapq.heappop(self.scheduled_drifts))
        return scheduled_reidentification

    def transition_reset(self, transition_timestep: int) -> None:
        """Resets the schedule on drift by clearing it, then adding any required new checks."""
        self.scheduled_drifts = []
        for scheduler in self.schedulers:
            new_checks = scheduler.transition_reset(transition_timestep)
            for check in new_checks:
                if check.drift_timestep <= transition_timestep:
                    continue
                heapq.heappush(self.scheduled_drifts, check)

    def initialize(self, initial_timestep: int) -> None:
        """Initialize schedulers."""
        self.scheduled_drifts = []
        for scheduler in self.schedulers:
            new_checks = scheduler.get_initialization_checks(initial_timestep)
            for check in new_checks:
                if check.drift_timestep <= initial_timestep:
                    continue
                heapq.heappush(self.scheduled_drifts, check)
