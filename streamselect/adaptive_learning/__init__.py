""" Adaptive learning systems which perform concept drift detection and concept
re-identification to adapt to concept drift in a streaming setting.

An adaptive learning system has three main tasks, with an overall goal of
maintaining an active state which is relevant to current streaming conditions.

1) Monitor the relevance of the current active state to recent data. (Concept drift detection.)
2) Calculate the relevance of inactive states from previous concepts to recent data. (Concept re-identification.)
3) Construct the optimal active state to handle new observations (Concept adaptation.)
"""

from .base import BaseAdaptiveLearner

__all__ = ["BaseAdaptiveLearner"]
