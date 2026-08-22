"""
Trajectory Quality Assessment: Smoothness, Missing Ratios, Physical Plausibility, and Integrity.
"""

from __future__ import annotations

import numpy as np

from traffic_intelligence.schema import Trajectory


class TrajectoryQualityEvaluator:
    """Evaluates the reliability, continuity, and physical realism of reconstructed trajectories."""

    @staticmethod
    def evaluate(
        trajectory: Trajectory,
        min_length_frames: int = 10,
        max_speed_kmh: float = 200.0,
        max_accel_mps2: float = 12.0,
    ) -> Trajectory:
        """Computes summary statistics and quality score for a trajectory."""
        pts = trajectory.points
        N = len(pts)

        if N < min_length_frames:
            trajectory.is_valid = False
            trajectory.quality_score = 0.2
            return trajectory

        # Frame continuity and missing ratio
        expected_frames = max(1, trajectory.end_frame - trajectory.start_frame + 1)
        missing_frames = max(0, expected_frames - N)
        trajectory.missing_ratio = float(missing_frames / expected_frames)

        # Kinematics stats
        speeds_kmh = [p.speed_kmh for p in pts if p.speed_kmh is not None]
        accels = [p.acceleration_magnitude_mps2 for p in pts if p.acceleration_magnitude_mps2 is not None]
        jerks = [p.jerk_mps3 for p in pts if p.jerk_mps3 is not None]

        if speeds_kmh:
            trajectory.average_speed_kmh = float(np.mean(speeds_kmh))
            trajectory.max_speed_kmh = float(np.max(speeds_kmh))
            # Dwell time (speed < 3 km/h)
            dwell_pts = sum(1 for s in speeds_kmh if s < 3.0)
            trajectory.dwell_time_s = float(dwell_pts * (trajectory.duration_s / max(1, N)))

        if accels:
            pos_acc = [a for a in accels if a > 0]
            neg_acc = [a for a in accels if a < 0]
            trajectory.max_acceleration_mps2 = float(np.max(pos_acc)) if pos_acc else 0.0
            trajectory.max_deceleration_mps2 = float(np.min(neg_acc)) if neg_acc else 0.0

        # Total distance
        coords = trajectory.get_world_coordinates_array()
        if len(coords) > 1:
            diffs = np.diff(coords, axis=0)
            trajectory.total_distance_m = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))

        # Quality scoring penalties
        penalties = 0.0
        # 1. Missing ratio penalty
        penalties += trajectory.missing_ratio * 0.4

        # 2. Impossible speed violation penalty
        if trajectory.max_speed_kmh > max_speed_kmh:
            penalties += 0.3

        # 3. Excessive jerk penalty (jittery unphysical tracker)
        if jerks and np.mean(np.abs(jerks)) > 30.0:
            penalties += 0.2

        # 4. Too few frames
        if N < 15:
            penalties += 0.1

        score = max(0.0, min(1.0, 1.0 - penalties))
        trajectory.quality_score = float(score)
        trajectory.is_valid = score >= 0.40

        return trajectory
