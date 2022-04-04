from typing import Dict, Tuple


class TransitionCounter:
    """Tracks transition weights."""

    def __init__(self) -> None:
        self.direct_weight: float = 0
        self.indirect_weight: float = 0
        self.total_weight: float = 0

    def add_direct(self, weight: float) -> None:
        """Add Direct transition between two states."""
        self.direct_weight += weight
        self.total_weight += weight

    def rem_direct(self, weight: float) -> None:
        """Remove Direct transition between two states."""
        self.direct_weight -= weight
        self.total_weight -= weight

    def add_indirect(self, weight: float) -> None:
        """Add Indirect transition, i.e., mediated by some deleted state."""
        self.indirect_weight += weight
        self.total_weight += weight

    def rem_indirect(self, weight: float) -> None:
        """Remove Indirect transition, i.e., mediated by some deleted state."""
        self.indirect_weight -= weight
        self.total_weight -= weight


class TransitionFSM:
    """Describes the transitions between states, or active state history.
    Can be used to calculate state priors. Uses an adjacency list internally.
    NOTE: States should have unique state_ids."""

    def __init__(self) -> None:
        self.adjacency_list: Dict[int, Dict[int, TransitionCounter]] = {}
        self.out_degrees: Dict[int, TransitionCounter] = {}
        self.in_degrees: Dict[int, TransitionCounter] = {}

    def add_transition(self, from_state_id: int, to_state_id: int, weight: int = 1) -> None:
        """Record a transition from a state with given id to another state with given id."""
        from_transitions = self.adjacency_list.setdefault(from_state_id, {})
        from_transitions.setdefault(to_state_id, TransitionCounter()).add_direct(weight)
        self.out_degrees.setdefault(from_state_id, TransitionCounter()).add_direct(weight)
        self.in_degrees.setdefault(to_state_id, TransitionCounter()).add_direct(weight)

        # Init empty counters
        self.adjacency_list.setdefault(to_state_id, {})
        self.out_degrees.setdefault(to_state_id, TransitionCounter())
        self.in_degrees.setdefault(from_state_id, TransitionCounter())

    def get_transition_weight(self, from_state_id: int, to_state_id: int, smoothing_weight: int = 0) -> float:
        """Get the weight associated with a given transition. If not found, return the min value."""
        try:
            return smoothing_weight + self.adjacency_list[from_state_id][to_state_id].total_weight
        except KeyError:
            return smoothing_weight

    def get_mle_prev_state(self, to_state_id: int) -> Tuple[int, float]:
        """Returns the id of the most likely previous state, and number of transitions.
        If no transitions exist, returns itself."""
        possible_previous_states = [(0.0, to_state_id)]
        for from_state_id, from_transitions in self.adjacency_list.items():
            if to_state_id in from_transitions:
                possible_previous_states.append((from_transitions[to_state_id].total_weight, from_state_id))

        prev_state_count, prev_state_id = max(possible_previous_states, key=lambda x: x[0])
        return prev_state_id, prev_state_count

    def delete_state(self, del_state_id: int, retain_indirect_weight: bool = True) -> None:
        """Delete a state from the FSM.
        If retain_indirect_weight is set, we maintain weight between previous
        state and next states as indirect. Since we do not know how to assign this weight, it
        is proportional."""
        if del_state_id not in self.in_degrees:
            return

        total_inweight = self.in_degrees[del_state_id]
        del self.in_degrees[del_state_id]
        in_neighbor_proportions: Dict[int, float] = {}
        for in_neighbor_id, in_trans in self.adjacency_list.items():
            if del_state_id in in_trans:
                transition_weight = in_trans[del_state_id]
                in_neighbor_proportions[in_neighbor_id] = transition_weight.total_weight / total_inweight.total_weight

                # if we are simply deleting, we need to remove outweight associated with the deleted state
                # Otherwise, we transition direct weight to indirect
                del in_trans[del_state_id]
                self.out_degrees[in_neighbor_id].rem_direct(transition_weight.direct_weight)
                self.out_degrees[in_neighbor_id].rem_indirect(transition_weight.indirect_weight)
                if retain_indirect_weight:
                    self.out_degrees[in_neighbor_id].add_indirect(transition_weight.total_weight)

        del self.out_degrees[del_state_id]
        out_trans = self.adjacency_list.setdefault(del_state_id, {})
        for out_neighbor_id, transition_weight in out_trans.items():
            # if we are simply deleting, we need to remove outweight associated with the deleted state
            # Otherwise, we transition weight from the deletect state to indirect weight from each prev state
            self.in_degrees[out_neighbor_id].rem_direct(transition_weight.direct_weight)
            self.in_degrees[out_neighbor_id].rem_indirect(transition_weight.indirect_weight)
            if retain_indirect_weight:
                for in_neighbor_id, in_proportion in in_neighbor_proportions.items():
                    new_indirect_weight = in_proportion * transition_weight.total_weight
                    self.adjacency_list[in_neighbor_id].setdefault(out_neighbor_id, TransitionCounter()).add_indirect(
                        new_indirect_weight
                    )
                self.in_degrees[out_neighbor_id].add_indirect(transition_weight.total_weight)

        del self.adjacency_list[del_state_id]
