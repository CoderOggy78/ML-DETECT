"""
Class-Wise Traffic Statistics and Speed Distribution Calculator.
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd

from traffic_intelligence.schema import RoadUserClass, Trajectory
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.statistics")


class ClassWiseStatisticsCalculator:
    """Computes comprehensive velocity, volume, and acceleration distributions per road-user class."""

    @staticmethod
    def compute_summary_table(trajectories: List[Trajectory]) -> pd.DataFrame:
        if not trajectories:
            return pd.DataFrame()

        records = []
        by_class: Dict[RoadUserClass, List[Trajectory]] = {}
        for t in trajectories:
            by_class.setdefault(t.class_name, []).append(t)

        for cls_enum, traj_list in by_class.items():
            speeds = []
            durations = []
            distances = []
            max_accels = []
            max_decels = []

            for t in traj_list:
                durations.append(t.duration_s)
                distances.append(t.total_distance_m)
                max_accels.append(t.max_acceleration_mps2)
                max_decels.append(t.max_deceleration_mps2)
                s_list = [p.speed_kmh for p in t.points if p.speed_kmh is not None]
                speeds.extend(s_list)

            s_arr = np.array(speeds) if speeds else np.array([0.0])

            records.append(
                {
                    "class_name": cls_enum.value,
                    "count": len(traj_list),
                    "mean_speed_kmh": float(np.mean(s_arr)),
                    "median_speed_kmh": float(np.median(s_arr)),
                    "p85_speed_kmh": float(np.percentile(s_arr, 85)),
                    "max_speed_kmh": float(np.max(s_arr)),
                    "mean_duration_s": float(np.mean(durations)),
                    "mean_distance_m": float(np.mean(distances)),
                    "max_accel_mps2": float(np.max(max_accels)) if max_accels else 0.0,
                    "max_decel_mps2": float(np.min(max_decels)) if max_decels else 0.0,
                }
            )

        df = pd.DataFrame(records).sort_values(by="count", ascending=False)
        return df
