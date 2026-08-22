"""
Coordinate systems, spatial transformations, distance metrics, and heading angles.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union
import numpy as np


def pixel_to_normalized(
    pixel_x: float, pixel_y: float, image_width: int, image_height: int
) -> Tuple[float, float]:
    """Converts pixel coordinates to normalized [0, 1] range."""
    return (pixel_x / max(1, image_width), pixel_y / max(1, image_height))


def euclidean_distance_2d(p1: Union[Tuple[float, float], np.ndarray], p2: Union[Tuple[float, float], np.ndarray]) -> float:
    """Calculates 2D Euclidean distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def calculate_heading_deg(dx: float, dy: float) -> float:
    """
    Calculates movement heading angle in degrees [0, 360) clockwise from North (Y-axis upward)
    or standard Cartesian navigation angle.
    Standard convention: East = 0 deg, North = 90 deg, West = 180 deg, South = 270 deg.
    """
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return (angle_deg + 360.0) % 360.0


def calculate_angle_between_headings(heading1_deg: float, heading2_deg: float) -> float:
    """Calculates the absolute shortest angular difference between two headings in degrees [0, 180]."""
    diff = abs(heading1_deg - heading2_deg) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


class PixelToWorldTransformer:
    """Transforms 2D image pixel coordinates (u, v) into metric ground-plane world coordinates (X, Y)."""

    def __init__(
        self,
        homography_matrix: Optional[np.ndarray] = None,
        meters_per_pixel: Optional[float] = None,
        origin_pixel: Tuple[float, float] = (0.0, 0.0),
    ):
        self.homography_matrix = np.array(homography_matrix, dtype=np.float64) if homography_matrix is not None else None
        self.meters_per_pixel = meters_per_pixel
        self.origin_pixel = origin_pixel

    def transform_point(self, px: float, py: float) -> Tuple[float, float]:
        """Transforms a single (pixel_x, pixel_y) point into metric world meters (x_m, y_m)."""
        if self.homography_matrix is not None:
            vec = np.array([px, py, 1.0], dtype=np.float64)
            res = self.homography_matrix @ vec
            if abs(res[2]) > 1e-7:
                return (float(res[0] / res[2]), float(res[1] / res[2]))
            return (float(res[0]), float(res[1]))

        elif self.meters_per_pixel is not None:
            ox, oy = self.origin_pixel
            return (
                float((px - ox) * self.meters_per_pixel),
                float((py - oy) * self.meters_per_pixel),
            )

        else:
            # Identity / pixel space
            return (float(px), float(py))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transforms array of shape (N, 2) to world coordinates (N, 2)."""
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float64)

        if self.homography_matrix is not None:
            pts_hom = np.hstack([points, np.ones((len(points), 1), dtype=np.float64)])
            res = (self.homography_matrix @ pts_hom.T).T
            scale = np.where(np.abs(res[:, 2:3]) > 1e-7, res[:, 2:3], 1.0)
            return res[:, :2] / scale

        elif self.meters_per_pixel is not None:
            ox, oy = self.origin_pixel
            shifted = points - np.array([ox, oy], dtype=np.float64)
            return shifted * self.meters_per_pixel

        return points.copy()
