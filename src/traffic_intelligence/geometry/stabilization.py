"""
Drone Camera Motion Estimator and Video Stabilization using ORB/Optical Flow & RANSAC Homography.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from traffic_intelligence.geometry.homography import HomographyEstimator, warp_points
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.stabilization")


class CameraMotionEstimator:
    """Estimates inter-frame camera motion and computes cumulative homographies to stabilize aerial video."""

    def __init__(
        self,
        method: str = "orb_ransac",
        max_features: int = 1500,
        ransac_reproj_threshold: float = 3.0,
        smoothing_window: int = 15,
        fallback_to_identity: bool = True,
    ):
        self.method = method
        self.max_features = max_features
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.smoothing_window = smoothing_window
        self.fallback_to_identity = fallback_to_identity

        self._orb = cv2.ORB_create(nfeatures=max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_keypoints: Optional[List[cv2.KeyPoint]] = None
        self._prev_descriptors: Optional[np.ndarray] = None

        # Cumulative transformations: maps frame t back to reference frame 0: H_{0 <- t}
        self.cumulative_homographies: Dict[int, np.ndarray] = {0: np.eye(3, dtype=np.float64)}
        self.inter_frame_homographies: Dict[int, np.ndarray] = {}
        self.residual_errors: Dict[int, float] = {}

    def reset(self) -> None:
        """Resets tracking state for a new video sequence."""
        self._prev_gray = None
        self._prev_keypoints = None
        self._prev_descriptors = None
        self.cumulative_homographies = {0: np.eye(3, dtype=np.float64)}
        self.inter_frame_homographies = {}
        self.residual_errors = {}

    def estimate_motion(self, frame_id: int, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Estimates inter-frame homography H_{t-1 -> t} and returns cumulative H_{0 <- t}.
        """
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if len(frame_bgr.shape) == 3 else frame_bgr

        if self._prev_gray is None or frame_id == 0:
            self._prev_gray = curr_gray
            if self.method == "orb_ransac":
                self._prev_keypoints, self._prev_descriptors = self._orb.detectAndCompute(curr_gray, None)
            self.cumulative_homographies[frame_id] = np.eye(3, dtype=np.float64)
            return np.eye(3, dtype=np.float64)

        H_inter = np.eye(3, dtype=np.float64)
        error = 0.0

        if self.method == "orb_ransac":
            curr_kp, curr_desc = self._orb.detectAndCompute(curr_gray, None)
            if (
                self._prev_descriptors is not None
                and curr_desc is not None
                and len(self._prev_descriptors) >= 8
                and len(curr_desc) >= 8
            ):
                matches = self._matcher.knnMatch(self._prev_descriptors, curr_desc, k=2)
                # Lowe's ratio test
                good_matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]

                if len(good_matches) >= 6:
                    src_pts = np.float32([self._prev_keypoints[m.queryIdx].pt for m in good_matches])
                    dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in good_matches])

                    H, inliers = HomographyEstimator.estimate_from_points(
                        src_pts, dst_pts, cv2.RANSAC, self.ransac_reproj_threshold
                    )
                    if H is not None:
                        H_inter = H
                        if inliers is not None and np.sum(inliers) > 0:
                            inlier_mask = inliers.ravel() == 1
                            src_inliers = src_pts[inlier_mask]
                            dst_inliers = dst_pts[inlier_mask]
                            warped = warp_points(src_inliers, H)
                            error = float(np.mean(np.linalg.norm(warped - dst_inliers, axis=1)))
                    else:
                        H_affine = HomographyEstimator.estimate_affine_fallback(src_pts, dst_pts)
                        if H_affine is not None:
                            H_inter = H_affine

            self._prev_keypoints = curr_kp
            self._prev_descriptors = curr_desc

        else:
            # Sparse Optical Flow fallback
            p0 = cv2.goodFeaturesToTrack(self._prev_gray, maxCorners=500, qualityLevel=0.01, minDistance=10)
            if p0 is not None and len(p0) >= 4:
                p1, st, err = cv2.calcOpticalFlowPyrLK(self._prev_gray, curr_gray, p0, None)
                if p1 is not None and st is not None:
                    good_p0 = p0[st == 1]
                    good_p1 = p1[st == 1]
                    if len(good_p0) >= 4:
                        H, inliers = HomographyEstimator.estimate_from_points(
                            good_p0, good_p1, cv2.RANSAC, self.ransac_reproj_threshold
                        )
                        if H is not None:
                            H_inter = H

        self._prev_gray = curr_gray
        self.inter_frame_homographies[frame_id] = H_inter
        self.residual_errors[frame_id] = error

        # Cumulative: H_{0 <- t} = H_{0 <- t-1} @ inv(H_{t-1 -> t})
        prev_cum = self.cumulative_homographies.get(frame_id - 1, np.eye(3, dtype=np.float64))
        try:
            H_inter_inv = np.linalg.inv(H_inter)
            cum_H = prev_cum @ H_inter_inv
        except np.linalg.LinAlgError:
            cum_H = prev_cum

        self.cumulative_homographies[frame_id] = cum_H
        return cum_H

    def compensate_coordinates(self, frame_id: int, coords_xy: np.ndarray) -> np.ndarray:
        """Warps points from frame t's image coordinate system to reference frame 0 coordinate system."""
        H_cum = self.cumulative_homographies.get(frame_id, np.eye(3, dtype=np.float64))
        return warp_points(coords_xy, H_cum)
