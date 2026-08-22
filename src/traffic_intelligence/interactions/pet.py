"""
Surrogate Safety Metric: Post-Encroachment Time (PET) calculation.
Measures time interval between first road user vacating a spatial zone and second user arriving.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from traffic_intelligence.schema import Trajectory


class PostEncroachmentTimeCalculator:
    """Calculates PET by discretizing spatial conflict regions into a spatial-temporal occupancy grid."""

    def __init__(self, cell_size_m: float = 2.0):
        self.cell_size_m = cell_size_m

    def compute_pairwise_pet(
        self, traj_a: Trajectory, traj_b: Trajectory, max_pet_threshold_s: float = 5.0
    ) -> Optional[float]:
        """Computes minimum PET between two complete trajectories in seconds."""
        # Check temporal overlap or proximity
        if (
            traj_a.end_timestamp_s < traj_b.start_timestamp_s - max_pet_threshold_s
            or traj_b.end_timestamp_s < traj_a.start_timestamp_s - max_pet_threshold_s
        ):
            return None

        # Build spatial-temporal occupancy map for traj_a: cell -> timestamp
        grid_a: Dict[Tuple[int, int], float] = {}
        for p in traj_a.points:
            wx = p.world_x_m if p.world_x_m is not None else p.pixel_x
            wy = p.world_y_m if p.world_y_m is not None else p.pixel_y
            cx = int(np.floor(wx / self.cell_size_m))
            cy = int(np.floor(wy / self.cell_size_m))
            grid_a[(cx, cy)] = p.timestamp_s

        min_pet: Optional[float] = None

        # Query traj_b points against traj_a occupancy
        for p in traj_b.points:
            wx = p.world_x_m if p.world_x_m is not None else p.pixel_x
            wy = p.world_y_m if p.world_y_m is not None else p.pixel_y
            cx = int(np.floor(wx / self.cell_size_m))
            cy = int(np.floor(wy / self.cell_size_m))

            if (cx, cy) in grid_a:
                t_a = grid_a[(cx, cy)]
                t_b = p.timestamp_s
                pet = abs(t_b - t_a)
                if pet <= max_pet_threshold_s:
                    if min_pet is None or pet < min_pet:
                        min_pet = pet

        return min_pet
