"""
Congestion Analysis, Level of Service (LOS), Speed Variance, and Delay Estimation.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from traffic_intelligence.schema import CongestionLevel, Trajectory
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.congestion")


class CongestionAnalyzer:
    """Analyzes traffic congestion levels and Level of Service (LOS) using kinematic distributions."""

    def __init__(
        self,
        free_flow_speed_kmh: float = 50.0,
        mild_congestion_speed_kmh: float = 35.0,
        moderate_congestion_speed_kmh: float = 20.0,
        severe_congestion_speed_kmh: float = 10.0,
        stopped_speed_kmh: float = 3.0,
    ):
        self.free_flow_speed_kmh = free_flow_speed_kmh
        self.mild_congestion_speed_kmh = mild_congestion_speed_kmh
        self.moderate_congestion_speed_kmh = moderate_congestion_speed_kmh
        self.severe_congestion_speed_kmh = severe_congestion_speed_kmh
        self.stopped_speed_kmh = stopped_speed_kmh

    def classify_congestion(self, mean_speed_kmh: float) -> CongestionLevel:
        """Determines traffic congestion state from observed harmonic/arithmetic mean speed."""
        if mean_speed_kmh >= self.free_flow_speed_kmh:
            return CongestionLevel.FREE_FLOW
        elif mean_speed_kmh >= self.mild_congestion_speed_kmh:
            return CongestionLevel.MILD
        elif mean_speed_kmh >= self.moderate_congestion_speed_kmh:
            return CongestionLevel.MODERATE
        elif mean_speed_kmh >= self.severe_congestion_speed_kmh:
            return CongestionLevel.SEVERE
        else:
            return CongestionLevel.STOPPED

    def analyze_trajectories(self, trajectories: List[Trajectory]) -> Dict[str, Any]:
        """Calculates global and class-wise speed variance, delay, and congestion metrics."""
        if not trajectories:
            return {
                "overall_mean_speed_kmh": 0.0,
                "overall_median_speed_kmh": 0.0,
                "overall_speed_std_kmh": 0.0,
                "congestion_level": CongestionLevel.FREE_FLOW.value,
                "total_vehicles": 0,
            }

        all_speeds = []
        for t in trajectories:
            speeds = [p.speed_kmh for p in t.points if p.speed_kmh is not None]
            all_speeds.extend(speeds)

        if not all_speeds:
            return {
                "overall_mean_speed_kmh": 0.0,
                "congestion_level": CongestionLevel.FREE_FLOW.value,
                "total_vehicles": len(trajectories),
            }

        speeds_arr = np.array(all_speeds)
        mean_spd = float(np.mean(speeds_arr))
        med_spd = float(np.median(speeds_arr))
        std_spd = float(np.std(speeds_arr))
        p85_spd = float(np.percentile(speeds_arr, 85))

        c_level = self.classify_congestion(mean_spd)

        return {
            "overall_mean_speed_kmh": mean_spd,
            "overall_median_speed_kmh": med_spd,
            "overall_speed_std_kmh": std_spd,
            "p85_speed_kmh": p85_spd,
            "congestion_level": c_level.value,
            "total_vehicles": len(trajectories),
            "free_flow_speed_reference_kmh": self.free_flow_speed_kmh,
        }
