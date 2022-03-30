""" A class to buffer arriving observations before they are learned from.
In a setting where concept drift may occur, it is dangerous to learn directly
from new observations, as this may cause us to learn from multiple distributions.
A buffer is a simple way to wait a number of timesteps before learning. """

from collections import deque
from typing import Dict, List, Optional

from river.base.typing import ClfTarget


class Observation:
    """A container class for a stored observation."""

    def __init__(self, X: dict, y: Optional[ClfTarget], sample_weight: float, seen_at: float) -> None:
        self.X = X
        self.y = y
        self.sample_weight = sample_weight
        self.seen_at = seen_at
        # A mappting between state_ids and their predictions for this observation.
        self.predictions: Dict[int, ClfTarget] = {}

    def add_prediction(self, p: ClfTarget, state_id: int) -> None:
        self.predictions[state_id] = p


class ObservationBuffer:
    """A buffer to store observations before learning from."""

    def __init__(self, window_size: int, buffer_timeout: float) -> None:
        """
        Parameters
        ----------
        window_size: int
            The number of observations to calculate representations over.

        buffer_timeout: float
            The number of timesteps to buffer observations.
            Can be updated, i.e., buffer less early then slow down.
        """
        self.window_size = window_size
        self.buffer_timeout = buffer_timeout

        self.buffer: deque[Observation] = deque()
        self.active_window: deque[Observation] = deque(maxlen=window_size)
        self.stable_window: deque[Observation] = deque(maxlen=window_size)

    def buffer_data(
        self, X: dict, y: Optional[ClfTarget], sample_weight: float, current_timestamp: float
    ) -> List[Observation]:
        """Add X and y to the buffer.
        y is optional, and should be set to None if not known.
        Current_timestamp is a float corresponding to the current timestamp (could be the data index).
        Returns all observations released from the buffer."""
        return self.add_observation(Observation(X, y, sample_weight, current_timestamp), current_timestamp)

    def add_observation(self, observation: Observation, current_timestamp: float) -> List[Observation]:
        """Add a new observation.
        Added directly to the active window, and to the buffer.
        When observation.seen_at is older than the buffer_timeout, it is released
        to the stable window.
        Returns all observations released from the buffer."""
        self.active_window.append(observation)
        self.buffer.append(observation)

        return self.release_buffer(current_timestamp)

    def release_buffer(self, current_timestamp: float) -> List[Observation]:
        """Release and return all observations from the buffer older than buffer_timeout."""
        released = []
        while current_timestamp - self.buffer[0].seen_at >= self.buffer_timeout:
            released.append(self.buffer.popleft())

        return released
