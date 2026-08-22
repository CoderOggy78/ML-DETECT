"""
8-State Kalman Filter for Multi-Object Tracking with Camera Motion Compensation (CMC).
State vector: [center_x, center_y, aspect_ratio, height, vx, vy, va, vh].
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from scipy.linalg import block_diag

from traffic_intelligence.geometry.homography import warp_points


class KalmanBoxTracker:
    """Kalman Filter estimating bounding box position, aspect ratio, height, and velocities."""

    count = 0

    def __init__(
        self,
        bbox_xyxy: Tuple[float, float, float, float],
        std_weight_position: float = 0.05,
        std_weight_velocity: float = 0.00625,
    ):
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count

        self.std_weight_position = std_weight_position
        self.std_weight_velocity = std_weight_velocity

        # State transition matrix F (8x8) for constant velocity model
        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix H (4x8)
        self.H = np.eye(4, 8, dtype=np.float64)

        # Initialize state mean x (8,) and covariance P (8x8)
        self.mean = np.zeros(8, dtype=np.float64)
        self.mean[:4] = self._bbox_to_z(bbox_xyxy)
        self.covariance = np.eye(8, dtype=np.float64) * 10.0

        self.age = 0
        self.time_since_update = 0
        self.hit_streak = 1

    @staticmethod
    def _bbox_to_z(bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Converts [xmin, ymin, xmax, ymax] to [cx, cy, aspect_ratio (w/h), height]."""
        w = max(1.0, bbox[2] - bbox[0])
        h = max(1.0, bbox[3] - bbox[1])
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        return np.array([cx, cy, w / h, h], dtype=np.float64)

    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> Tuple[float, float, float, float]:
        """Converts [cx, cy, aspect_ratio, height] back to [xmin, ymin, xmax, ymax]."""
        cx, cy, a, h = z[0], z[1], max(1e-4, z[2]), max(1.0, z[3])
        w = a * h
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    def _get_process_noise_Q(self) -> np.ndarray:
        h = self.mean[3]
        std_pos = [
            self.std_weight_position * h,
            self.std_weight_position * h,
            1e-2,
            self.std_weight_position * h,
        ]
        std_vel = [
            self.std_weight_velocity * h,
            self.std_weight_velocity * h,
            1e-5,
            self.std_weight_velocity * h,
        ]
        diag = np.square(np.r_[std_pos, std_vel])
        return np.diag(diag)

    def _get_measurement_noise_R(self) -> np.ndarray:
        h = self.mean[3]
        std = [
            self.std_weight_position * h,
            self.std_weight_position * h,
            1e-1,
            self.std_weight_position * h,
        ]
        return np.diag(np.square(std))

    def predict(self, cmc_matrix: Optional[np.ndarray] = None) -> Tuple[float, float, float, float]:
        """
        Advances state by 1 timestep and applies camera motion compensation if provided.
        """
        # Apply camera motion transformation to spatial mean if camera shifted
        if cmc_matrix is not None:
            # Warp center coordinate
            pt = np.array([[self.mean[0], self.mean[1]]], dtype=np.float64)
            warped_pt = warp_points(pt, cmc_matrix)[0]
            self.mean[0], self.mean[1] = warped_pt[0], warped_pt[1]

            # Scale height by homography zoom factor
            scale = np.sqrt(cmc_matrix[0, 0] ** 2 + cmc_matrix[0, 1] ** 2)
            if scale > 0.1:
                self.mean[3] *= scale

        # Standard Kalman Predict: x' = F x, P' = F P F^T + Q
        Q = self._get_process_noise_Q()
        self.mean = self.F @ self.mean
        self.covariance = self.F @ self.covariance @ self.F.T + Q

        self.age += 1
        self.time_since_update += 1
        return self._z_to_bbox(self.mean[:4])

    def update(self, bbox_xyxy: Tuple[float, float, float, float]) -> None:
        """
        Kalman measurement update with new bounding box observation.
        """
        z = self._bbox_to_z(bbox_xyxy)
        R = self._get_measurement_noise_R()

        # Innovation: y = z - H x
        y = z - self.H @ self.mean
        # Innovation covariance: S = H P H^T + R
        S = self.H @ self.covariance @ self.H.T + R
        # Kalman Gain: K = P H^T S^-1
        K = self.covariance @ self.H.T @ np.linalg.inv(S)

        # Updated state and covariance: x = x + K y, P = (I - K H) P
        self.mean = self.mean + K @ y
        I = np.eye(8, dtype=np.float64)
        self.covariance = (I - K @ self.H) @ self.covariance

        self.time_since_update = 0
        self.hit_streak += 1

    def get_state_bbox(self) -> Tuple[float, float, float, float]:
        """Returns current bounding box estimate."""
        return self._z_to_bbox(self.mean[:4])

    def get_velocity_xy(self) -> Tuple[float, float]:
        """Returns estimated (vx, vy) in pixels/frame."""
        return (float(self.mean[4]), float(self.mean[5]))
