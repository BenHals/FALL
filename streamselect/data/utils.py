from river.datasets.base import Dataset


class Concept:
    def __init__(self, data: Dataset, name: str, max_n: int = -1):
        """A container class representing a data stream concept.
        Contains a dataset which will be used to draw observations whenever this concept is
        active.
        Parameters
        ---------
        data: Dataset
            The underlying data which observations will be drawn.
            Can be real or synthetic.

        name: str
            Should be a unique human readable name representing the concept.

        max_n: int
            The max number of observations to take before repeating.
            Set to -1 for no repeats until data is exhausted."""
        self.data = data
        self.name = name
        self.max_n = max_n


class ConceptSegment:
    def __init__(self, concept_stream: Concept, segment_start: int, segment_end: int, recurrence_count: int):
        """A container class representing a concept segment."""
        self.concept = concept_stream
        self.segment_start = segment_start
        self.segment_end = segment_end
        self.recurrence_count = recurrence_count
