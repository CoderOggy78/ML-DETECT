"""
Lane Model and Unsupervised Lane Centerline Estimator from Trajectory Densities.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from sklearn.cluster import DBSCAN

from traffic_intelligence.schema import Trajectory
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.lanes")


class LaneModel:
    """Represents a discrete traffic lane with centerline coordinates and width."""

    def __init__(self, lane_id: str, centerline_points: np.ndarray, width_m: float = 3.5):
        self.lane_id = lane_id
        self.centerline_points = centerline_points
        self.width_m = width_m

    def distance_to_point(self, pt: Tuple[float, float]) -> float:
        if len(self.centerline_points) == 0:
            return float("inf")
        diffs = self.centerline_points - np.array(pt)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        return float(np.min(dists))


class LaneEstimator:
    """Estimates dominant lane centerlines from trajectory flow lines without manual road markings."""

    @staticmethod
    def estimate_lanes_from_trajectories(
        trajectories: List[Trajectory], eps_m: float = 3.0, min_samples: int = 3
    ) -> List[LaneModel]:
        if not trajectories:
            return []

        # Sample points along moving trajectories
        sampled_points = []
        for traj in trajectories:
            if traj.average_speed_kmh > 5.0:
                coords = traj.get_world_coordinates_array()
                if len(coords) > 5:
                    step = max(1, len(coords) // 10)
                    sampled_points.extend(coords[::step])

        if len(sampled_points) < min_samples:
            return []

        pts_arr = np.array(sampled_points, dtype=np.float64)
        db = DBSCAN(eps=eps_m, min_samples=min_samples).fit(pts_arr)
        labels = db.labels_

        lanes: List[LaneModel] = []
        unique_labels = set(labels) - {-1}

        for idx, lbl in enumerate(sorted(unique_labels)):
            cluster_pts = pts_arr[labels == lbl]
            if len(cluster_pts) > 5:
                # Sort along primary PCA axis
                mean = np.mean(cluster_pts, axis=0)
                uu, dd, vv = np.linalg.svd(cluster_pts - mean)
                proj = (cluster_pts - mean) @ vv[0]
                sorted_pts = cluster_pts[np.argsort(proj)]

                lanes.append(LaneModel(lane_id=f"Lane_{idx+1}", centerline_points=sorted_pts))

        logger.info(f"Discovered {len(lanes)} dominant lane corridors from trajectory clusters.")
        return lanes
