"""
Temporal Trend Analysis and Rolling Time-Window Aggregations (1m, 5m, 15m, 1h).
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from traffic_intelligence.schema import ConflictEvent, TrafficEvent, Trajectory
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.temporal")


class TemporalTrendAggregator:
    """Aggregates flow, average speed, conflict frequency, and events across time windows."""

    @staticmethod
    def aggregate_windowed_metrics(
        trajectories: List[Trajectory],
        conflicts: List[ConflictEvent],
        events: List[TrafficEvent],
        window_size_s: float = 60.0,
    ) -> pd.DataFrame:
        if not trajectories:
            return pd.DataFrame()

        t_min = min(t.start_timestamp_s for t in trajectories)
        t_max = max(t.end_timestamp_s for t in trajectories)

        windows = np.arange(t_min, t_max + window_size_s, window_size_s)
        rows = []

        for i in range(len(windows) - 1):
            w_start, w_end = windows[i], windows[i + 1]

            # Trajectories active in window
            w_trajs = [t for t in trajectories if not (t.end_timestamp_s < w_start or t.start_timestamp_s > w_end)]
            # Speeds in window
            speeds = []
            for t in w_trajs:
                for p in t.points:
                    if w_start <= p.timestamp_s < w_end and p.speed_kmh is not None:
                        speeds.append(p.speed_kmh)

            mean_spd = float(np.mean(speeds)) if speeds else 0.0

            # Conflicts in window
            w_conflicts = [c for c in conflicts if w_start <= c.timestamp_s < w_end]
            # Events in window
            w_events = [e for e in events if w_start <= e.start_timestamp_s < w_end]

            rows.append(
                {
                    "window_start_s": w_start,
                    "window_end_s": w_end,
                    "active_vehicle_count": len(w_trajs),
                    "mean_speed_kmh": mean_spd,
                    "conflict_count": len(w_conflicts),
                    "event_count": len(w_events),
                }
            )

        return pd.DataFrame(rows)
