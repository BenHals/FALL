""" Base state class"""

from river.base import Classifier

from streamselect.concept_representations import ConceptRepresentation


class State:  # pylint: disable=too-few-public-methods
    """A base state containing a Classifier and ConceptRepresentation"""

    def __init__(self, classifier: Classifier, concept_representation: ConceptRepresentation):
        self.classifier = classifier
        self.concept_representation = concept_representation
