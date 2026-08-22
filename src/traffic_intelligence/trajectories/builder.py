"""
Trajectory Builder: Assembles, transforms, smooths, and enriches raw track observations.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from traffic_intelligence.geometry.coordinates import PixelToWorldTransformer
from traffic_intelligence.schema import RoadUserClass, Track, Trajectory, TrajectoryPoint
from traffic_intelligence.trajectories.motion import MotionEstimator
from traffic_intelligence.trajectories.quality import TrajectoryQualityEvaluator
from traffic_intelligence.trajectories.smoothing import TrajectorySmoother
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.builder")


class TrajectoryBuilder:
    """Builds smoothed, physical trajectory models from raw tracking streams."""

    def __init__(
        self,
        transformer: Optional[PixelToWorldTransformer] = None,
        smoothing_method: str = "kalman",
        min_track_length: int = 10,
        savgol_window: int = 11,
        savgol_polyorder: int = 3,
        kalman_process_noise: float = 0.1,
        kalman_measurement_noise: float = 1.0,
        max_speed_mps: float = 60.0,
        max_accel_mps2: float = 15.0,
    ):
        self.transformer = transformer or PixelToWorldTransformer(meters_per_pixel=0.05)
        self.smoothing_method = smoothing_method
        self.min_track_length = min_track_length
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self.max_speed_mps = max_speed_mps
        self.max_accel_mps2 = max_accel_mps2

        self._track_observations: Dict[int, List[Track]] = defaultdict(list)

    def add_tracks(self, tracks: List[Track]) -> None:
        """Adds a list of track observations from a single frame."""
        for t in tracks:
            self._track_observations[t.track_id].append(t)

    def build_trajectories(self) -> List[Trajectory]:
        """Processes all accumulated observations and constructs finalized Trajectory objects."""
        trajectories: List[Trajectory] = []

        for track_id, observations in self._track_observations.items():
            if len(observations) < self.min_track_length:
                continue

            # Sort temporally by frame_id
            obs_sorted = sorted(observations, key=lambda t: t.frame_id)
            cls_name = self._majority_class([t.class_name for t in obs_sorted])

            # Extract raw pixel center coordinates
            pixel_centers = np.array([t.center_xy for t in obs_sorted], dtype=np.float64)

            # Apply smoothing
            smoothed_pixels = TrajectorySmoother.smooth(
                pixel_centers,
                method=self.smoothing_method,
                savgol_window=self.savgol_window,
                savgol_poly=self.savgol_polyorder,
                kalman_q=self.kalman_process_noise,
                kalman_r=self.kalman_measurement_noise,
            )

            # Transform to metric world coordinates
            world_coords = self.transformer.transform_points(smoothed_pixels)

            # Build TrajectoryPoint instances
            points: List[TrajectoryPoint] = []
            for i, obs in enumerate(obs_sorted):
                px_x, px_y = float(smoothed_pixels[i, 0]), float(smoothed_pixels[i, 1])
                wld_x, wld_y = float(world_coords[i, 0]), float(world_coords[i, 1])

                pt = TrajectoryPoint(
                    frame_id=obs.frame_id,
                    timestamp_s=obs.timestamp_s,
                    pixel_x=px_x,
                    pixel_y=px_y,
                    bbox_xyxy=obs.bbox_xyxy,
                    world_x_m=wld_x,
                    world_y_m=wld_y,
                    confidence=obs.confidence,
                )
                points.append(pt)

            # Compute kinematics (velocity, acceleration, jerk, heading)
            enriched_points = MotionEstimator.estimate_kinematics(
                points, max_speed_mps=self.max_speed_mps, max_accel_mps2=self.max_accel_mps2
            )

            traj = Trajectory(
                track_id=track_id,
                class_name=cls_name,
                start_frame=obs_sorted[0].frame_id,
                end_frame=obs_sorted[-1].frame_id,
                start_timestamp_s=obs_sorted[0].timestamp_s,
                end_timestamp_s=obs_sorted[-1].timestamp_s,
                points=enriched_points,
            )

            # Run quality validation
            traj = TrajectoryQualityEvaluator.evaluate(traj, min_length_frames=self.min_track_length)
            trajectories.append(traj)

        logger.info(f"Built {len(trajectories)} trajectories from {len(self._track_observations)} tracklets")
        return trajectories

    @staticmethod
    def _majority_class(classes: List[RoadUserClass]) -> RoadUserClass:
        if not classes:
            return RoadUserClass.UNKNOWN
        from collections import Counter
        counts = Counter(classes)
        return counts.most_common(1)[0][0]

    @staticmethod
    def trajectories_to_dataframe(trajectories: List[Trajectory]) -> pd.DataFrame:
        """Converts a list of Trajectory objects into a flattened pandas DataFrame for Parquet export."""
        rows = []
        for traj in trajectories:
            for p in traj.points:
                rows.append(
                    {
                        "track_id": traj.track_id,
                        "class_name": traj.class_name.value,
                        "frame_id": p.frame_id,
                        "timestamp_s": p.timestamp_s,
                        "pixel_x": p.pixel_x,
                        "pixel_y": p.pixel_y,
                        "world_x_m": p.world_x_m,
                        "world_y_m": p.world_y_m,
                        "bbox_xmin": p.bbox_xyxy[0],
                        "bbox_ymin": p.bbox_xyxy[1],
                        "bbox_xmax": p.bbox_xyxy[2],
                        "bbox_ymax": p.bbox_xyxy[3],
                        "velocity_x_mps": p.velocity_x_mps,
                        "velocity_y_mps": p.velocity_y_mps,
                        "speed_mps": p.speed_mps,
                        "speed_kmh": p.speed_kmh,
                        "acceleration_mps2": p.acceleration_magnitude_mps2,
                        "jerk_mps3": p.jerk_mps3,
                        "heading_deg": p.heading_deg,
                        "angular_velocity_degps": p.angular_velocity_degps,
                        "movement_type": traj.movement_type.value,
                        "origin_zone": traj.origin_zone,
                        "destination_zone": traj.destination_zone,
                        "quality_score": traj.quality_score,
                    }
                )
        return pd.DataFrame(rows)
