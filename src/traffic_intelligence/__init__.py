"""
Traffic Intelligence: Aerial Traffic Intelligence & Trajectory Discovery Platform.
"""

__version__ = "1.0.0"
__author__ = "Traffic Intelligence Research Team"

from traffic_intelligence.schema import (
    Detection,
    Track,
    Trajectory,
    TrafficEvent,
    ConflictEvent,
    QueueState,
    RoadUserClass,
    SeverityLevel,
    MovementType,
    CongestionLevel
)

__all__ = [
    "Detection",
    "Track",
    "Trajectory",
    "TrafficEvent",
    "ConflictEvent",
    "QueueState",
    "RoadUserClass",
    "SeverityLevel",
    "MovementType",
    "CongestionLevel",
]
