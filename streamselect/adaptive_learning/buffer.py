""" A class to buffer arriving observations before they are learned from.
In a setting where concept drift may occur, it is dangerous to learn directly
from new observations, as this may cause us to learn from multiple distributions.
A buffer is a simple way to wait a number of timesteps before learning. """

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from river.base.typing import ClfTarget


class Observation:
    """A container class for a stored observation."""

    def __init__(self, x: dict, y: Optional[ClfTarget], sample_weight: float, seen_at: float) -> None:
        self.x = x
        self.y = y
        self.sample_weight = sample_weight
        self.seen_at = seen_at
        # A mappting between state_ids and their predictions for this observation.
        self.predictions: Dict[int, ClfTarget] = {}

    def add_prediction(self, p: ClfTarget, state_id: int) -> None:
        self.predictions[state_id] = p

    def __str__(self) -> str:
        return f"<{self.x}|{self.y}|{self.seen_at}>"

    def __repr__(self) -> str:
        return str(self)


class ObservationBuffer:
    """A buffer to store observations before learning from."""

    def __init__(self, window_size: int) -> None:
        """
        Parameters
        ----------
        window_size: int
            The number of observations to calculate representations over.
        """
        self.window_size = window_size

        self.buffer: Deque[Observation] = deque()  # pylint: disable=unsubscriptable-object
        self.active_window: Deque[Observation] = deque(maxlen=window_size)  # pylint: disable=unsubscriptable-object
        self.stable_window: Deque[Observation] = deque(maxlen=window_size)  # pylint: disable=unsubscriptable-object

    def buffer_data(
        self, x: dict, y: Optional[ClfTarget], sample_weight: float, current_timestamp: float, stable_timestamp: float
    ) -> List[Observation]:
        """Add x and y to the buffer.
        y is optional, and should be set to None if not known.
        Current_timestamp is a float corresponding to the current timestamp (could be the data index).
        Returns all observations released from the buffer."""
        return self.add_observation(Observation(x, y, sample_weight, current_timestamp), stable_timestamp)

    def add_observation(self, observation: Observation, stable_timestamp: float) -> List[Observation]:
        """Add a new observation.
        Added directly to the active window, and to the buffer.
        When observation.seen_at is older than the stable_timestamp, it is released
        to the stable window.
        Returns all observations released from the buffer."""
        self.active_window.append(observation)
        self.buffer.append(observation)

        return self.release_buffer(stable_timestamp)

    def release_buffer(self, stable_timestamp: float) -> List[Observation]:
        """Release and return all observations from the buffer older than stable_timestamp."""
        released = []
        while len(self.buffer) and self.buffer[0].seen_at <= stable_timestamp:
            stable_observation = self.buffer.popleft()
            released.append(stable_observation)
            self.stable_window.append(stable_observation)

        return released

    def reset_on_drift(self, drift_timestep: float = -1.0) -> None:
        """When a drift occurs, we must assume that data in the buffer before the drift_timestep is not stable
        for the new active state and should be removed."""

        while len(self.buffer) and self.buffer[0].seen_at <= drift_timestep:
            self.buffer.popleft()
        while len(self.active_window) and self.active_window[0].seen_at <= drift_timestep:
            self.active_window.popleft()
        while len(self.stable_window) and self.stable_window[0].seen_at <= drift_timestep:
            self.stable_window.popleft()


class SupervisedUnsupervisedBuffer:
    """A class for maintaining supervised and unsupervised data separately."""

    def __init__(
        self,
        window_size: int,
        unsupervised_buffer_timeout: float,
        supervised_buffer_timeout: float,
        release_strategy: str = "supervised",
    ) -> None:
        """Maintains two buffers for supervised and unsupervised data.
        Parameters
        ----------
        window_size: int
            The size of window to maintain for active and stable windows

        unsupervised_buffer_timeout: float
            Number of timesteps to buffer unsupervised observations, dependent on strategy.

        supervised_buffer_timeout: float
            Number of timesteps to buffer supervised observations, dependent on strategy.

        release_strategy: str [supervised, unsupervised, independent]
            How buffer release is handled. "independent" releases buffers independently
            depending on timestamp, while supervised and unsupervised release both buffers
            when the respective timeout passes. Dependent modes are used when drift detection
            is tied to a specific mode, i.e., only on supervised data.

        Notes
        -----
        We match observations by timestamp, i.e., a supervised observation with timestamp
        t should be associated with the unsupervised observation with timestamp t.
        For example, in "supervised" mode, if we identify timestep t as being stable
        due to observing supervised_buffer_timeout observations since t was received,
        we will unbuffer both supervised and unsupervised observations <= t.

        When specifically passed current_timestamp, we can handle missing data.
        This matching is done automatically by river's simulate_qa.
        We can also handle simple incrementing timestamps by setting current_timestamp to -1.0.
        However, the incremental strategy cannot handle missing data, as we cannot tell what
        timestamp a supervised observation represents.

        """
        self.window_size = window_size
        self.unsupervised_buffer_timeout = unsupervised_buffer_timeout
        self.supervised_buffer_timeout = supervised_buffer_timeout
        self.release_strategy = release_strategy

        self.supervised_active_timestamp = 0.0
        self.unsupervised_active_timestamp = 0.0

        self.unsupervised_buffer = ObservationBuffer(self.window_size)
        self.supervised_buffer = ObservationBuffer(self.window_size)

        self.stable_unsupervised: List[Observation] = []
        self.stable_supervised: List[Observation] = []

    def get_unsupervised_timestamps(self) -> Tuple[float, float]:
        """Returns the active timestamp and stable timestamp for unsupervised data.
        The active timestamp is what new data is added as, while the stable timestamp
        is the threshold for releasing data.

        If the release strategy is independent or unsupervised, this is based on unsupervised time.
        If supervised, we time and release data based on the supervised clock."""
        if self.release_strategy in ["independent", "unsupervised"]:
            return (
                self.unsupervised_active_timestamp,
                self.unsupervised_active_timestamp - self.unsupervised_buffer_timeout,
            )
        return self.supervised_active_timestamp, self.supervised_active_timestamp - self.supervised_buffer_timeout

    def get_supervised_timestamps(self) -> Tuple[float, float]:
        """Returns the active timestamp and stable timestamp for supervised data.
        The active timestamp is what new data is added as, while the stable timestamp
        is the threshold for releasing data.

        If the release strategy is independent or supervised, this is based on supervised time.
        If supervised, we time and release data based on the unsupervised clock."""
        if self.release_strategy in ["independent", "supervised"]:
            return self.supervised_active_timestamp, self.supervised_active_timestamp - self.supervised_buffer_timeout
        return (
            self.unsupervised_active_timestamp,
            self.unsupervised_active_timestamp - self.unsupervised_buffer_timeout,
        )

    def buffer_unsupervised(self, x: dict, current_timestamp: float = -1.0, sample_weight: float = 1.0) -> None:
        """Add an unsupervised observation.
        Current timestamp is -1.0 if unknown, in which case we simply increment by one.
        This is fine when data comes in at regular intervals.
        Otherwise, we set the timestamp to current_timestamp."""
        if current_timestamp == -1.0:
            self.unsupervised_active_timestamp += 1.0
        else:
            self.unsupervised_active_timestamp = current_timestamp

        _, stable_timestamp = self.get_unsupervised_timestamps()
        self.stable_unsupervised += self.unsupervised_buffer.buffer_data(
            x=x,
            y=None,
            sample_weight=sample_weight,
            current_timestamp=self.unsupervised_active_timestamp,
            stable_timestamp=stable_timestamp,
        )

    def buffer_supervised(
        self, x: dict, y: ClfTarget, current_timestamp: float = -1.0, sample_weight: float = 1.0
    ) -> None:
        """Add a supervised observation.
        Current timestamp is -1.0 if unknown, in which case we simply increment by one.
        This is fine when data comes in at regular intervals, but may not be accurate with missing
        or delayed data.
        Otherwise, we set the timestamp to current_timestamp."""
        if current_timestamp == -1.0:
            self.supervised_active_timestamp += 1.0
        else:
            self.supervised_active_timestamp = current_timestamp

        _, stable_timestamp = self.get_supervised_timestamps()
        self.stable_supervised += self.supervised_buffer.buffer_data(
            x=x,
            y=y,
            sample_weight=sample_weight,
            current_timestamp=self.supervised_active_timestamp,
            stable_timestamp=stable_timestamp,
        )

    def collect_stable_unsupervised(self) -> List[Observation]:
        """Get collected stable unsupervised data, and clear it from cache."""
        _, stable_timestamp = self.get_unsupervised_timestamps()
        self.stable_unsupervised += self.unsupervised_buffer.release_buffer(stable_timestamp)
        collected = self.stable_unsupervised
        self.stable_unsupervised = []
        return collected

    def collect_stable_supervised(self) -> List[Observation]:
        """Get collected stable supervised data, and clear it from cache."""
        _, stable_timestamp = self.get_supervised_timestamps()
        self.stable_supervised += self.supervised_buffer.release_buffer(stable_timestamp)
        collected = self.stable_supervised
        self.stable_supervised = []
        return collected

    def reset_on_drift(self, drift_timestep: float = -1.0) -> None:
        """When a drift occurs, we must assume that data in the buffer before the drift_timestep is not stable
        for the new active state. If drift timestep is not set, we assume it is window_size
        prior to the current timestep, i.e., the active window is stable."""
        unsupervised_drift_timestep = drift_timestep
        active_timestamp, stable_timestamp = self.get_unsupervised_timestamps()
        if unsupervised_drift_timestep == -1.0:
            unsupervised_drift_timestep = active_timestamp - self.window_size
        remaining_stable = int(stable_timestamp - drift_timestep)
        if remaining_stable > 0:
            self.stable_unsupervised = self.stable_unsupervised[-remaining_stable:]
        else:
            self.stable_unsupervised = []

        supervised_drift_timestep = drift_timestep
        active_timestamp, stable_timestamp = self.get_supervised_timestamps()
        if supervised_drift_timestep == -1.0:
            supervised_drift_timestep = active_timestamp - self.window_size
        remaining_stable = int(stable_timestamp - drift_timestep)
        if remaining_stable > 0:
            self.stable_supervised = self.stable_supervised[-remaining_stable:]
        else:
            self.stable_supervised = []

        self.unsupervised_buffer.reset_on_drift(unsupervised_drift_timestep)
        self.supervised_buffer.reset_on_drift(supervised_drift_timestep)
