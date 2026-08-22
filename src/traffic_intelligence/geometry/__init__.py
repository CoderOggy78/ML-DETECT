"""
Geometry, coordinate transformations, planar homography, camera calibration, and stabilization.
"""

from traffic_intelligence.geometry.coordinates import (
    PixelToWorldTransformer,
    pixel_to_normalized,
    euclidean_distance_2d,
    calculate_heading_deg,
    calculate_angle_between_headings,
)
from traffic_intelligence.geometry.homography import HomographyEstimator, warp_points
from traffic_intelligence.geometry.calibration import CalibrationManager
from traffic_intelligence.geometry.stabilization import CameraMotionEstimator

__all__ = [
    "PixelToWorldTransformer",
    "pixel_to_normalized",
    "euclidean_distance_2d",
    "calculate_heading_deg",
    "calculate_angle_between_headings",
    "HomographyEstimator",
    "warp_points",
    "CalibrationManager",
    "CameraMotionEstimator",
]
