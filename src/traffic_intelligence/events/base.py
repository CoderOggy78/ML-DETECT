"""
Abstract Base Event Detector and standardized TrafficEvent factory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional
from traffic_intelligence.schema import SeverityLevel, TrafficEvent, Trajectory


class BaseEventDetector(ABC):
    """Abstract interface for all behavioral event detectors."""

    @abstractmethod
    def detect(self, trajectories: List[Trajectory]) -> List[TrafficEvent]:
        """Scans trajectory corpus and returns detected domain traffic events."""
        pass
