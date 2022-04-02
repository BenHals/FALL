from typing import Dict, Optional

from river.base.typing import ClfTarget


class Observation:
    """A container class for a stored observation."""

    def __init__(
        self, x: dict, y: Optional[ClfTarget], seen_at: float, active_state_id: int, sample_weight: float = 1.0
    ) -> None:
        self.x = x
        self.y = y
        self.seen_at = seen_at
        self.active_state_id = active_state_id
        self.sample_weight = sample_weight
        # A mappting between state_ids and their predictions for this observation.
        self.predictions: Dict[int, ClfTarget] = {}

    def add_prediction(self, p: ClfTarget, state_id: int) -> None:
        self.predictions[state_id] = p

    def __str__(self) -> str:
        return f"<{self.x}|{self.y}|{self.seen_at}>"

    def __repr__(self) -> str:
        return str(self)
