"""
Geometric Movement Classification (Straight, Left Turn, Right Turn, U-Turn, Stopped)
and Intersection Turning Movement Matrices.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from traffic_intelligence.schema import MovementType, Trajectory
from traffic_intelligence.traffic.road_model import RoadModel
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.movements")


class MovementClassifier:
    """Classifies road-user trajectories into physical turning movements and builds turning matrices."""

    @staticmethod
    def classify_trajectory_movement(
        trajectory: Trajectory, road_model: Optional[RoadModel] = None
    ) -> MovementType:
        """
        Infers movement type geometrically from angular deflection, cumulative curvature, and net displacement.
        """
        pts = trajectory.get_world_coordinates_array()
        if len(pts) < 3:
            return MovementType.UNKNOWN

        # Check if stopped / stationary
        if trajectory.average_speed_kmh < 3.0 and trajectory.total_distance_m < 5.0:
            return MovementType.STOPPED

        # 1. Compute initial heading vs final heading
        start_vec = pts[min(5, len(pts) - 1)] - pts[0]
        end_vec = pts[-1] - pts[max(0, len(pts) - 6)]

        angle_start = math.degrees(math.atan2(start_vec[1], start_vec[0]))
        angle_end = math.degrees(math.atan2(end_vec[1], end_vec[0]))

        # Net heading change: counter-clockwise positive, clockwise negative
        delta_angle = (angle_end - angle_start + 180.0) % 360.0 - 180.0

        # Also check cumulative angle change along path
        headings = [p.heading_deg for p in trajectory.points if p.heading_deg is not None]
        total_turn = 0.0
        if len(headings) > 2:
            dh = np.diff(headings)
            dh_wrapped = (dh + 180.0) % 360.0 - 180.0
            total_turn = float(np.sum(dh_wrapped))

        # Classification based on turning angle:
        # Standard convention:
        # Straight: |delta| < 35 deg
        # Left Turn: +35 deg to +140 deg (or counter-clockwise)
        # Right Turn: -35 deg to -140 deg (or clockwise)
        # U-Turn: |delta| > 140 deg
        if abs(delta_angle) > 140.0 or abs(total_turn) > 150.0:
            return MovementType.U_TURN
        elif delta_angle > 35.0 or total_turn > 45.0:
            return MovementType.LEFT_TURN
        elif delta_angle < -35.0 or total_turn < -45.0:
            return MovementType.RIGHT_TURN
        else:
            return MovementType.STRAIGHT

    @classmethod
    def process_all_trajectories(
        cls, trajectories: List[Trajectory], road_model: Optional[RoadModel] = None
    ) -> List[Trajectory]:
        """Classifies all trajectories and attaches origin/destination zones."""
        for t in trajectories:
            t.movement_type = cls.classify_trajectory_movement(t, road_model)
            if road_model:
                origin, dest = road_model.identify_entry_exit(t)
                t.origin_zone = origin
                t.destination_zone = dest
        return trajectories

    @staticmethod
    def generate_turning_movement_matrix(trajectories: List[Trajectory]) -> pd.DataFrame:
        """Aggregates turning movements by origin, destination, and movement type."""
        records = []
        for t in trajectories:
            records.append(
                {
                    "track_id": t.track_id,
                    "class_name": t.class_name.value,
                    "movement_type": t.movement_type.value,
                    "origin_zone": t.origin_zone or "Unknown_Origin",
                    "destination_zone": t.destination_zone or "Unknown_Destination",
                }
            )

        if not records:
            return pd.DataFrame(columns=["origin_zone", "destination_zone", "movement_type", "count"])

        df = pd.DataFrame(records)
        grouped = (
            df.groupby(["origin_zone", "destination_zone", "movement_type"])
            .size()
            .reset_index(name="count")
            .sort_values(by="count", ascending=False)
        )
        return grouped
