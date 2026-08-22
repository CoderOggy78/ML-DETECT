"""
Spatial Heatmap Generation: 2D Kernel Density Estimation and Grid Rasters for Density, Speed, Dwell, and Conflicts.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from scipy.ndimage import gaussian_filter

from traffic_intelligence.schema import ConflictEvent, Trajectory
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.spatial")


class SpatialHeatmapGenerator:
    """Generates normalized 2D spatial grid rasters and heatmaps from trajectory coordinates."""

    def __init__(self, grid_size: Tuple[int, int] = (100, 100)):
        self.grid_rows, self.grid_cols = grid_size

    def _get_coordinate_bounds(self, trajectories: List[Trajectory]) -> Tuple[float, float, float, float]:
        all_pts = []
        for t in trajectories:
            pts = t.get_world_coordinates_array()
            if len(pts) > 0:
                all_pts.append(pts)
        if not all_pts:
            return 0.0, 100.0, 0.0, 100.0

        stacked = np.vstack(all_pts)
        min_x, min_y = float(np.min(stacked[:, 0])), float(np.min(stacked[:, 1]))
        max_x, max_y = float(np.max(stacked[:, 0])), float(np.max(stacked[:, 1]))
        return min_x, max_x, min_y, max_y

    def generate_density_heatmap(
        self, trajectories: List[Trajectory], sigma: float = 2.0
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Generates 2D trajectory passage density raster."""
        min_x, max_x, min_y, max_y = self._get_coordinate_bounds(trajectories)
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float64)

        dx = max(1e-4, max_x - min_x)
        dy = max(1e-4, max_y - min_y)

        for t in trajectories:
            pts = t.get_world_coordinates_array()
            for p in pts:
                c = int(np.clip((p[0] - min_x) / dx * (self.grid_cols - 1), 0, self.grid_cols - 1))
                r = int(np.clip((p[1] - min_y) / dy * (self.grid_rows - 1), 0, self.grid_rows - 1))
                grid[r, c] += 1.0

        blurred = gaussian_filter(grid, sigma=sigma)
        max_val = np.max(blurred)
        normalized = blurred / max_val if max_val > 0 else blurred
        return normalized, (min_x, max_x, min_y, max_y)

    def generate_speed_heatmap(
        self, trajectories: List[Trajectory], sigma: float = 2.0
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Generates 2D mean speed spatial heatmap."""
        min_x, max_x, min_y, max_y = self._get_coordinate_bounds(trajectories)
        sum_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float64)
        count_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float64)

        dx = max(1e-4, max_x - min_x)
        dy = max(1e-4, max_y - min_y)

        for t in trajectories:
            for p in t.points:
                wx = p.world_x_m if p.world_x_m is not None else p.pixel_x
                wy = p.world_y_m if p.world_y_m is not None else p.pixel_y
                spd = p.speed_kmh or 0.0

                c = int(np.clip((wx - min_x) / dx * (self.grid_cols - 1), 0, self.grid_cols - 1))
                r = int(np.clip((wy - min_y) / dy * (self.grid_rows - 1), 0, self.grid_rows - 1))

                sum_grid[r, c] += spd
                count_grid[r, c] += 1.0

        avg_grid = np.divide(sum_grid, count_grid, out=np.zeros_like(sum_grid), where=count_grid > 0)
        blurred = gaussian_filter(avg_grid, sigma=sigma)
        return blurred, (min_x, max_x, min_y, max_y)

    def generate_conflict_heatmap(
        self, conflicts: List[ConflictEvent], trajectories: List[Trajectory], sigma: float = 2.5
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Generates 2D spatial hotspot raster of surrogate safety conflicts."""
        min_x, max_x, min_y, max_y = self._get_coordinate_bounds(trajectories)
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float64)

        dx = max(1e-4, max_x - min_x)
        dy = max(1e-4, max_y - min_y)

        for c in conflicts:
            x, y = c.location_world or c.location_pixel
            col = int(np.clip((x - min_x) / dx * (self.grid_cols - 1), 0, self.grid_cols - 1))
            row = int(np.clip((y - min_y) / dy * (self.grid_rows - 1), 0, self.grid_rows - 1))
            weight = 3.0 if c.severity.value == "CRITICAL" else (2.0 if c.severity.value == "HIGH" else 1.0)
            grid[row, col] += weight

        blurred = gaussian_filter(grid, sigma=sigma)
        max_val = np.max(blurred)
        normalized = blurred / max_val if max_val > 0 else blurred
        return normalized, (min_x, max_x, min_y, max_y)
