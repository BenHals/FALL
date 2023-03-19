"""
Contains tools for loading, constructing and handling data streams with concept drift.
In particular, we focus on handling recurring concept drift.
"""
from .datastream import ConceptSegmentDataStream

__all__ = ["ConceptSegmentDataStream"]
