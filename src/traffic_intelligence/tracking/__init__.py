"""
Multi-Object Tracking (ByteTrack, BoT-SORT), Kalman Filter, Re-Identification, and Track Lifecycle Management.
"""

from traffic_intelligence.tracking.base import BaseTracker
from traffic_intelligence.tracking.kalman import KalmanBoxTracker
from traffic_intelligence.tracking.reid import BaseReID, AppearanceExtractor
from traffic_intelligence.tracking.bytetrack import ByteTracker
from traffic_intelligence.tracking.botsort import BoTSORTTracker
from traffic_intelligence.tracking.track_manager import TrackManager

__all__ = [
    "BaseTracker",
    "KalmanBoxTracker",
    "BaseReID",
    "AppearanceExtractor",
    "ByteTracker",
    "BoTSORTTracker",
    "TrackManager",
]
