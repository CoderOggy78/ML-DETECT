"""
Trajectory Feature Extraction: 24+ statistical, geometric, and kinematic descriptors per trajectory.
"""

from __future__ import annotations

import math
from typing import List
import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, skew

from traffic_intelligence.schema import Trajectory


class TrajectoryFeatureExtractor:
    """Extracts fixed-dimensional invariant feature vectors for unsupervised discovery."""

    FEATURE_NAMES = [
        "duration_s",
        "total_distance_m",
        "net_displacement_m",
        "tortuosity_ratio",
        "mean_speed_mps",
        "median_speed_mps",
        "std_speed_mps",
        "max_speed_mps",
        "p85_speed_mps",
        "mean_accel_mps2",
        "max_accel_mps2",
        "max_decel_mps2",
        "accel_skewness",
        "accel_kurtosis",
        "mean_jerk_mps3",
        "jerk_energy",
        "total_turning_angle_deg",
        "mean_angular_velocity_degps",
        "max_angular_velocity_degps",
        "heading_entropy",
        "dwell_time_ratio",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
    ]

    @classmethod
    def extract_features_single(cls, trajectory: Trajectory) -> np.ndarray:
        """Computes 25-dimensional feature vector for a single trajectory."""
        pts = trajectory.points
        N = len(pts)
        if N < 2:
            return np.zeros(len(cls.FEATURE_NAMES), dtype=np.float64)

        # Coordinates
        w_pts = trajectory.get_world_coordinates_array()
        start_pt = w_pts[0]
        end_pt = w_pts[-1]
        net_disp = float(np.hypot(end_pt[0] - start_pt[0], end_pt[1] - start_pt[1]))
        tot_dist = max(1e-3, trajectory.total_distance_m or net_disp)
        tortuosity = float(tot_dist / max(1e-3, net_disp))

        # Speed features
        speeds = np.array([p.speed_mps or 0.0 for p in pts], dtype=np.float64)
        mean_spd = float(np.mean(speeds))
        med_spd = float(np.median(speeds))
        std_spd = float(np.std(speeds))
        max_spd = float(np.max(speeds))
        p85_spd = float(np.percentile(speeds, 85))

        # Acceleration features
        accels = np.array([p.acceleration_magnitude_mps2 or 0.0 for p in pts], dtype=np.float64)
        mean_acc = float(np.mean(accels))
        pos_acc = accels[accels > 0]
        neg_acc = accels[accels < 0]
        max_acc = float(np.max(pos_acc)) if len(pos_acc) > 0 else 0.0
        max_dec = float(np.min(neg_acc)) if len(neg_acc) > 0 else 0.0
        acc_skew = float(skew(accels)) if len(accels) > 3 and std_spd > 1e-4 else 0.0
        acc_kurt = float(kurtosis(accels)) if len(accels) > 3 and std_spd > 1e-4 else 0.0

        # Jerk features
        jerks = np.array([p.jerk_mps3 or 0.0 for p in pts], dtype=np.float64)
        mean_jerk = float(np.mean(np.abs(jerks)))
        jerk_energy = float(np.sum(jerks ** 2) / max(1, N))

        # Heading & Turning features
        headings = [p.heading_deg for p in pts if p.heading_deg is not None]
        total_turn = 0.0
        head_entropy = 0.0
        if len(headings) > 2:
            dh = np.diff(headings)
            dh_wrapped = (dh + 180.0) % 360.0 - 180.0
            total_turn = float(np.sum(np.abs(dh_wrapped)))

            # Discrete histogram for entropy
            hist, _ = np.histogram(headings, bins=8, range=(0, 360), density=True)
            hist = hist[hist > 0]
            head_entropy = float(entropy(hist)) if len(hist) > 0 else 0.0

        ang_vels = np.array([abs(p.angular_velocity_degps or 0.0) for p in pts], dtype=np.float64)
        mean_ang_vel = float(np.mean(ang_vels))
        max_ang_vel = float(np.max(ang_vels))

        # Dwell ratio
        dwell_pts = sum(1 for s in speeds if s < 0.8)
        dwell_ratio = float(dwell_pts / max(1, N))

        feat_vec = [
            trajectory.duration_s,
            tot_dist,
            net_disp,
            tortuosity,
            mean_spd,
            med_spd,
            std_spd,
            max_spd,
            p85_spd,
            mean_acc,
            max_acc,
            max_dec,
            acc_skew,
            acc_kurt,
            mean_jerk,
            jerk_energy,
            total_turn,
            mean_ang_vel,
            max_ang_vel,
            head_entropy,
            dwell_ratio,
            start_pt[0],
            start_pt[1],
            end_pt[0],
            end_pt[1],
        ]

        # Clean NaNs / Infs
        clean = np.nan_to_num(np.array(feat_vec, dtype=np.float64), nan=0.0, posinf=1e4, neginf=-1e4)
        return clean

    @classmethod
    def extract_feature_matrix(cls, trajectories: List[Trajectory]) -> pd.DataFrame:
        """Extracts full feature matrix for a batch of trajectories."""
        rows = []
        for t in trajectories:
            vec = cls.extract_features_single(t)
            row_dict = {"track_id": t.track_id, "class_name": t.class_name.value}
            row_dict.update({name: val for name, val in zip(cls.FEATURE_NAMES, vec)})
            rows.append(row_dict)

        return pd.DataFrame(rows)
