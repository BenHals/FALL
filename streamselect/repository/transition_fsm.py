from typing import Dict


class TransitionFSM:
    """Describes the transitions between states, or active state history.
    Can be used to calculate state priors. Uses an adjacency list internally.
    NOTE: States should have unique state_ids."""

    def __init__(self) -> None:
        self.adjacency_list: Dict[int, Dict[int, int]] = {}
        self.out_degrees: Dict[int, int] = {}
        self.in_degrees: Dict[int, int] = {}

    def add_transition(self, from_state_id: int, to_state_id: int, weight: int = 1) -> None:
        """Record a transition from a state with given id to another state with given id."""
        from_transitions = self.adjacency_list.setdefault(from_state_id, {})
        from_transitions[to_state_id] = from_transitions.get(to_state_id, 0) + weight
        self.out_degrees[from_state_id] = self.out_degrees.get(from_state_id, 0) + weight
        self.in_degrees[to_state_id] = self.in_degrees.get(to_state_id, 0) + weight

    def get_transition_weight(self, from_state_id: int, to_state_id: int, smoothing_weight: int = 0) -> int:
        """Get the weight associated with a given transition. If not found, return the min value."""
        try:
            return smoothing_weight + self.adjacency_list[from_state_id][to_state_id]
        except KeyError:
            return smoothing_weight
