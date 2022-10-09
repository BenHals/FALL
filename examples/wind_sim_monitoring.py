from river.drift.adwin import ADWIN
from river.naive_bayes import GaussianNB

from streamselect.adaptive_learning import BaseAdaptiveLearner
from streamselect.concept_representations import ErrorRateRepresentation
from streamselect.data.datastream import ConceptSegmentDataStream, make_stream_concepts
from streamselect.data.synthetic.wind_sim import WindSimGenerator
from streamselect.data.transition_patterns import circular_transition_pattern
from streamselect.data.utils import Concept
from streamselect.evaluation.monitoring import Monitor
from streamselect.repository import AbsoluteValueComparer

s0 = WindSimGenerator(concept=0)
s1 = WindSimGenerator(concept=1)
s2 = WindSimGenerator(concept=2)
s3 = WindSimGenerator(concept=3)
c0 = Concept(s0, "0")
c1 = Concept(s1, "1")
c2 = Concept(s2, "2")
c3 = Concept(s3, "3")

n_repeats = 30
pattern = circular_transition_pattern(4, n_repeats, 1.0, 1, 0.0, shuffle_order=False)

segment_length = 1000
concept_segments = make_stream_concepts([c0, c1, c2, c3], pattern, segment_length)

seed = 42
datastream = ConceptSegmentDataStream(concept_segments, 0, seed)

classifier = BaseAdaptiveLearner(
    classifier_constructor=GaussianNB,
    representation_constructor=ErrorRateRepresentation,
    representation_comparer=AbsoluteValueComparer(),
    drift_detector_constructor=ADWIN,
    representation_window_size=50,
    representation_update_period=5,
    drift_detection_mode="lower",
)

baseline = GaussianNB()

if __name__ == "__main__":
    print(pattern)
    print(concept_segments)
    print(datastream)

    monitor = Monitor()
    monitor.run_monitor(datastream, classifier, baseline)
