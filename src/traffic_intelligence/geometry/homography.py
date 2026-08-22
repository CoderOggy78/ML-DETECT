"""
Planar Homography estimation using RANSAC and Direct Linear Transformation (DLT).
"""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np

from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.homography")


def warp_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warps 2D points using 3x3 homography matrix H."""
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)

    pts_hom = np.hstack([points, np.ones((len(points), 1), dtype=np.float64)])
    warped = (H @ pts_hom.T).T
    scale = np.where(np.abs(warped[:, 2:3]) > 1e-7, warped[:, 2:3], 1.0)
    return warped[:, :2] / scale


class HomographyEstimator:
    """Estimates projective transformations between frame pairs or image-to-ground coordinates."""

    @staticmethod
    def estimate_from_points(
        src_points: np.ndarray,
        dst_points: np.ndarray,
        method: int = cv2.RANSAC,
        ransac_reproj_threshold: float = 3.0,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Computes 3x3 homography matrix H such that dst ~ H @ src.
        Requires at least 4 corresponding point pairs.
        """
        if len(src_points) < 4 or len(dst_points) < 4:
            logger.warning("At least 4 correspondence points are required to compute Homography.")
            return None, None

        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)

        H, inliers = cv2.findHomography(src, dst, method, ransac_reproj_threshold)
        if H is not None and not HomographyEstimator.is_valid_homography(H):
            logger.warning("Estimated Homography matrix is numerically degenerate or ill-conditioned.")
            return None, None

        return H, inliers

    @staticmethod
    def estimate_affine_fallback(src_points: np.ndarray, dst_points: np.ndarray) -> Optional[np.ndarray]:
        """Estimates a 2x3 affine matrix and converts to 3x3 projective matrix."""
        if len(src_points) < 3:
            return None
        M, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
        if M is None:
            return None
        H = np.eye(3, dtype=np.float64)
        H[:2, :] = M
        return H

    @staticmethod
    def is_valid_homography(H: np.ndarray) -> bool:
        """Sanity checks condition number and determinant of homography matrix."""
        if H is None or H.shape != (3, 3):
            return False
        if np.any(np.isnan(H)) or np.any(np.isinf(H)):
            return False
        det = np.linalg.det(H)
        if abs(det) < 1e-6 or abs(det) > 1e8:
            return False
        return True
