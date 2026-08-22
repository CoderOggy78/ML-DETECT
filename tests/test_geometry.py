"""
Unit tests for geometry, coordinate transformation, homography, and stabilization.
"""

import numpy as np
import pytest
from traffic_intelligence.geometry.coordinates import (
    PixelToWorldTransformer,
    calculate_heading_deg,
    calculate_angle_between_headings,
    euclidean_distance_2d,
)
from traffic_intelligence.geometry.homography import HomographyEstimator, warp_points
from traffic_intelligence.geometry.stabilization import CameraMotionEstimator


def test_heading_and_angle_difference():
    # East (dx=1, dy=0) -> 0 deg
    assert calculate_heading_deg(1.0, 0.0) == pytest.approx(0.0)
    # North (dx=0, dy=1) -> 90 deg
    assert calculate_heading_deg(0.0, 1.0) == pytest.approx(90.0)
    # West (dx=-1, dy=0) -> 180 deg
    assert calculate_heading_deg(-1.0, 0.0) == pytest.approx(180.0)

    # Angle difference
    assert calculate_angle_between_headings(10.0, 350.0) == pytest.approx(20.0)
    assert calculate_angle_between_headings(0.0, 180.0) == pytest.approx(180.0)
    assert calculate_angle_between_headings(90.0, 45.0) == pytest.approx(45.0)


def test_pixel_to_world_transformer_scale():
    trans = PixelToWorldTransformer(meters_per_pixel=0.1, origin_pixel=(10.0, 10.0))
    wx, wy = trans.transform_point(110.0, 210.0)
    assert wx == pytest.approx(10.0)
    assert wy == pytest.approx(20.0)

    pts = np.array([[10.0, 10.0], [20.0, 30.0]])
    res = trans.transform_points(pts)
    assert res.shape == (2, 2)
    assert res[0, 0] == pytest.approx(0.0)
    assert res[0, 1] == pytest.approx(0.0)


def test_homography_estimation_and_warping():
    # Create known affine/projective mapping: x' = 2*x + 5, y' = 2*y + 10
    src = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]], dtype=np.float32)
    dst = np.array([[5.0, 10.0], [205.0, 10.0], [205.0, 210.0], [5.0, 210.0]], dtype=np.float32)

    H, inliers = HomographyEstimator.estimate_from_points(src, dst)
    assert H is not None
    assert HomographyEstimator.is_valid_homography(H)

    warped = warp_points(src, H)
    np.testing.assert_allclose(warped, dst, atol=1e-3)


def test_camera_motion_estimator():
    estimator = CameraMotionEstimator()
    frame1 = np.full((100, 100, 3), 120, dtype=np.uint8)
    frame2 = np.full((100, 100, 3), 120, dtype=np.uint8)

    H_cum = estimator.estimate_motion(0, frame1)
    assert H_cum.shape == (3, 3)
    np.testing.assert_allclose(H_cum, np.eye(3))

    H_cum2 = estimator.estimate_motion(1, frame2)
    assert H_cum2.shape == (3, 3)
