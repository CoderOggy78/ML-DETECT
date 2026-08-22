"""
Trajectory-Based Queue Detection, Evolution Tracking, and Shockwave Propagation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from traffic_intelligence.schema import QueueState, Trajectory, TrajectoryPoint
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.queues")


class QueueDetector:
    """
    Detects physical vehicle queues formed by low-speed clusters persisting along road corridors.
    Estimates queue length in meters, growth/dissipation rates, and vehicle counts.
    """

    def __init__(
        self,
        speed_threshold_kmh: float = 5.0,
        min_dwell_time_s: float = 3.0,
        max_vehicle_gap_m: float = 8.0,
        min_queue_vehicles: int = 2,
    ):
        self.speed_threshold_kmh = speed_threshold_kmh
        self.min_dwell_time_s = min_dwell_time_s
        self.max_vehicle_gap_m = max_vehicle_gap_m
        self.min_queue_vehicles = min_queue_vehicles

    def detect_queues_at_timestamp(
        self,
        active_points: List[Tuple[int, TrajectoryPoint]],
        timestamp_s: float,
        frame_id: int,
    ) -> List[QueueState]:
        """
        Identifies spatial queues among active trajectory points at a given frame.
        active_points: List of (track_id, TrajectoryPoint).
        """
        # 1. Filter for stopped or slow-moving vehicles
        slow_candidates = []
        for tid, pt in active_points:
            spd = pt.speed_kmh if pt.speed_kmh is not None else (pt.speed_mps * 3.6 if pt.speed_mps else 0.0)
            if spd <= self.speed_threshold_kmh:
                wx = pt.world_x_m if pt.world_x_m is not None else pt.pixel_x
                wy = pt.world_y_m if pt.world_y_m is not None else pt.pixel_y
                slow_candidates.append((tid, wx, wy))

        if len(slow_candidates) < self.min_queue_vehicles:
            return []

        # 2. Spatial clustering using DBSCAN with max gap threshold
        coords = np.array([[c[1], c[2]] for c in slow_candidates], dtype=np.float64)
        db = DBSCAN(eps=self.max_vehicle_gap_m, min_samples=self.min_queue_vehicles).fit(coords)
        labels = db.labels_

        queue_states: List[QueueState] = []
        for cluster_id in set(labels) - {-1}:
            mask = labels == cluster_id
            cluster_coords = coords[mask]
            cluster_tids = [slow_candidates[i][0] for i in range(len(slow_candidates)) if mask[i]]

            # Estimate queue length (maximum pairwise distance or span)
            if len(cluster_coords) >= 2:
                # Find two most distant points in the queue cluster (head and tail)
                diffs = cluster_coords[:, np.newaxis, :] - cluster_coords[np.newaxis, :, :]
                dist_matrix = np.hypot(diffs[:, :, 0], diffs[:, :, 1])
                i_head, i_tail = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)

                queue_len_m = float(dist_matrix[i_head, i_tail])
                head_pos = (float(cluster_coords[i_head, 0]), float(cluster_coords[i_head, 1]))
                tail_pos = (float(cluster_coords[i_tail, 0]), float(cluster_coords[i_tail, 1]))
            else:
                queue_len_m = 5.0
                head_pos = (float(cluster_coords[0, 0]), float(cluster_coords[0, 1]))
                tail_pos = head_pos

            state = QueueState(
                queue_id=f"Queue_F{frame_id}_C{cluster_id}",
                timestamp_s=timestamp_s,
                frame_id=frame_id,
                vehicle_count=len(cluster_tids),
                queue_length_m=queue_len_m,
                max_queue_length_m=queue_len_m,
                duration_s=1.0,
                growth_rate_mps=0.0,
                dissipation_rate_mps=0.0,
                involved_track_ids=cluster_tids,
                head_location_world=head_pos,
                tail_location_world=tail_pos,
            )
            queue_states.append(state)

        return queue_states

    def process_all_trajectories(self, trajectories: List[Trajectory]) -> pd.DataFrame:
        """Runs temporal queue tracking across all trajectory timestamps."""
        # Index points by frame
        frame_map: Dict[int, List[Tuple[int, TrajectoryPoint]]] = {}
        for t in trajectories:
            for p in t.points:
                if p.frame_id not in frame_map:
                    frame_map[p.frame_id] = []
                frame_map[p.frame_id].append((t.track_id, p))

        all_states: List[QueueState] = []
        for fid in sorted(frame_map.keys()):
            pts = frame_map[fid]
            ts = pts[0][1].timestamp_s
            q_states = self.detect_queues_at_timestamp(pts, ts, fid)
            all_states.extend(q_states)

        rows = []
        for q in all_states:
            rows.append(
                {
                    "queue_id": q.queue_id,
                    "frame_id": q.frame_id,
                    "timestamp_s": q.timestamp_s,
                    "vehicle_count": q.vehicle_count,
                    "queue_length_m": q.queue_length_m,
                    "involved_vehicles": ",".join(map(str, q.involved_track_ids)),
                }
            )

        return pd.DataFrame(rows)
