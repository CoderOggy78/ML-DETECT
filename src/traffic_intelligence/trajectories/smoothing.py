"""
Trajectory Smoothing using Rauch-Tung-Striebel (RTS) Kalman Smoothing and Savitzky-Golay Filters.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.signal import savgol_filter

from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.smoothing")


class TrajectorySmoother:
    """Applies filtering and noise attenuation to spatial trajectory series."""

    @staticmethod
    def savitzky_golay_smooth(
        coords: np.ndarray, window_length: int = 11, polyorder: int = 3
    ) -> np.ndarray:
        """
        Applies Savitzky-Golay polynomial smoothing to 2D trajectory coordinates.
        Preserves high-frequency inflection points (sudden turns, abrupt braking).
        """
        N = len(coords)
        if N < 4:
            return coords.copy()

        # Window length must be odd and <= N
        w = min(window_length, N if N % 2 != 0 else N - 1)
        if w < 3:
            w = 3
        poly = min(polyorder, w - 1)

        try:
            smoothed_x = savgol_filter(coords[:, 0], window_length=w, polyorder=poly)
            smoothed_y = savgol_filter(coords[:, 1], window_length=w, polyorder=poly)
            return np.column_stack([smoothed_x, smoothed_y])
        except Exception as e:
            logger.debug(f"Savitzky-Golay smoothing fallback: {e}")
            return coords.copy()

    @staticmethod
    def kalman_rts_smooth(
        coords: np.ndarray, process_noise: float = 0.1, measurement_noise: float = 1.0
    ) -> np.ndarray:
        """
        Rauch-Tung-Striebel (RTS) optimal forward-backward Kalman Smoother for 2D position trajectories.
        """
        N = len(coords)
        if N < 3:
            return coords.copy()

        dt = 1.0
        # State: [x, y, vx, vy]^T
        F = np.array(
            [[1.0, 0.0, dt, 0.0],
             [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        H = np.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        Q = np.eye(4, dtype=np.float64) * process_noise
        R = np.eye(2, dtype=np.float64) * measurement_noise

        # 1. Forward Pass (Standard Kalman Filter)
        x_pred = np.zeros((N, 4), dtype=np.float64)
        P_pred = np.zeros((N, 4, 4), dtype=np.float64)
        x_filt = np.zeros((N, 4), dtype=np.float64)
        P_filt = np.zeros((N, 4, 4), dtype=np.float64)

        # Initialize
        x_filt[0, :2] = coords[0]
        x_filt[0, 2:] = coords[1] - coords[0] if N > 1 else 0.0
        P_filt[0] = np.eye(4, dtype=np.float64) * 10.0

        for k in range(1, N):
            # Predict
            x_pred[k] = F @ x_filt[k - 1]
            P_pred[k] = F @ P_filt[k - 1] @ F.T + Q

            # Update
            z = coords[k]
            y = z - H @ x_pred[k]
            S = H @ P_pred[k] @ H.T + R
            K = P_pred[k] @ H.T @ np.linalg.inv(S)
            x_filt[k] = x_pred[k] + K @ y
            P_filt[k] = (np.eye(4) - K @ H) @ P_pred[k]

        # 2. Backward Pass (RTS Smoother)
        x_smooth = np.zeros_like(x_filt)
        x_smooth[-1] = x_filt[-1]

        for k in range(N - 2, -1, -1):
            C = P_filt[k] @ F.T @ np.linalg.inv(P_pred[k + 1])
            x_smooth[k] = x_filt[k] + C @ (x_smooth[k + 1] - x_pred[k + 1])

        return x_smooth[:, :2]

    @classmethod
    def smooth(
        cls,
        coords: np.ndarray,
        method: str = "kalman",
        savgol_window: int = 11,
        savgol_poly: int = 3,
        kalman_q: float = 0.1,
        kalman_r: float = 1.0,
    ) -> np.ndarray:
        if method == "savitzky_golay":
            return cls.savitzky_golay_smooth(coords, savgol_window, savgol_poly)
        elif method == "kalman":
            return cls.kalman_rts_smooth(coords, kalman_q, kalman_r)
        elif method == "none":
            return coords.copy()
        return cls.kalman_rts_smooth(coords, kalman_q, kalman_r)
