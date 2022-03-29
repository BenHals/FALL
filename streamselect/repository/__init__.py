""" Handles the creation of a repository containing previously built states.

A data stream can be considered as a sequence of concepts associated with some underlying environmental context.
Each concept may feature a distinct joint distribution of data, so should be learned by a distinct classifier.
When concepts may reoccur over time, we wish to retain these classifiers for reuse, essentially creating a
finite state machine.
The code in this module handles the mechanisms for representing each state, and the repository which stores
constructed states and their memory management.
"""

from .base import Repository

__all__ = ["Repository"]
