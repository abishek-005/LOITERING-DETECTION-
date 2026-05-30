"""
SENTINEL v2.0 — Core Detection & Tracking Package
===================================================
Re-exports the main classes so consumers can write::

    from core import PersonDetector, PersonTracker, PoseEstimator, LoiteringDetector
"""

from .detector import PersonDetector
from .tracker import PersonTracker
from .pose_estimator import PoseEstimator
from .loitering import LoiteringDetector

__all__ = [
    "PersonDetector",
    "PersonTracker",
    "PoseEstimator",
    "LoiteringDetector",
]
