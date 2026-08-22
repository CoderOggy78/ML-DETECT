"""
Traffic Flow Rate, Spatial Density, and Temporal Window Aggregator.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from traffic_intelligence.schema import Trajectory
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.flow")


class TrafficFlowEstimator:
    """Estimates flow rate (veh/hr), density (veh/km), and class-wise distributions across time bins."""

    @staticmethod
    def compute_binned_flow(
        trajectories: List[Trajectory], bin_size_s: float = 60.0
    ) -> pd.DataFrame:
        if not trajectories:
            return pd.DataFrame(columns=["bin_start_s", "bin_end_s", "vehicle_count", "flow_rate_veh_hr"])

        min_t = min(t.start_timestamp_s for t in trajectories)
        max_t = max(t.end_timestamp_s for t in trajectories)

        bins = np.arange(min_t, max_t + bin_size_s, bin_size_s)
        rows = []

        for i in range(len(bins) - 1):
            t0, t1 = bins[i], bins[i + 1]
            active_trajs = [
                t for t in trajectories
                if not (t.end_timestamp_s < t0 or t.start_timestamp_s > t1)
            ]
            count = len(active_trajs)
            flow_per_hr = count * (3600.0 / bin_size_s)

            # Class breakdown
            class_counts = {}
            for t in active_trajs:
                c = t.class_name.value
                class_counts[c] = class_counts.get(c, 0) + 1

            rows.append(
                {
                    "bin_start_s": t0,
                    "bin_end_s": t1,
                    "vehicle_count": count,
                    "flow_rate_veh_hr": flow_per_hr,
                    **class_counts,
                }
            )

        return pd.DataFrame(rows).fillna(0)
