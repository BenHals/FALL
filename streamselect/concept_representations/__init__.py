""" Base classes for concept representations.
A concept is a joint distribution between X and y.
A concept representation is a finite sized approximation of this distribution using a given classifier.
Each concept distribution should have a method of construction from a window of observations and
a similarity method to another concept representation. Ideally, it should also be able to be updated
online."""

from .base import ConceptRepresentation

__all__ = ["ConceptRepresentation"]
