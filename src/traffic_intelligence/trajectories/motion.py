"""
Physical Motion Estimation: Velocity, Speed, Acceleration, Jerk, Heading, and Angular Velocity.
Calculates derivatives in metric world coordinates with robust numerical differentiation.
"""

from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np

from traffic_intelligence.geometry.coordinates import calculate_heading_deg
from traffic_intelligence.schema import TrajectoryPoint


class MotionEstimator:
    """Computes high-order physical kinematic metrics along a trajectory."""

    @staticmethod
    def estimate_kinematics(
        points: List[TrajectoryPoint],
        max_speed_mps: float = 60.0,
        max_accel_mps2: float = 15.0,
    ) -> List[TrajectoryPoint]:
        """
        Computes velocity, speed, acceleration, jerk, heading, and angular velocity.
        Uses central differences for interior points and forward/backward differences at boundaries.
        """
        N = len(points)
        if N == 0:
            return points

        if N == 1:
            points[0].velocity_x_mps = 0.0
            points[0].velocity_y_mps = 0.0
            points[0].speed_mps = 0.0
            points[0].speed_kmh = 0.0
            points[0].acceleration_magnitude_mps2 = 0.0
            points[0].jerk_mps3 = 0.0
            points[0].heading_deg = 0.0
            points[0].angular_velocity_degps = 0.0
            return points

        timestamps = np.array([p.timestamp_s for p in points], dtype=np.float64)
        
        # Determine whether to compute in world meters or pixel coords
        has_world = points[0].world_x_m is not None and points[0].world_y_m is not None
        if has_world:
            coords = np.array([[p.world_x_m, p.world_y_m] for p in points], dtype=np.float64)
        else:
            coords = np.array([[p.pixel_x, p.pixel_y] for p in points], dtype=np.float64)

        # 1. Velocities (vx, vy)
        vx = np.zeros(N, dtype=np.float64)
        vy = np.zeros(N, dtype=np.float64)

        # Forward difference for first point
        dt0 = max(1e-4, timestamps[1] - timestamps[0])
        vx[0] = (coords[1, 0] - coords[0, 0]) / dt0
        vy[0] = (coords[1, 1] - coords[0, 1]) / dt0

        # Central difference for interior points
        for i in range(1, N - 1):
            dt = max(1e-4, timestamps[i + 1] - timestamps[i - 1])
            vx[i] = (coords[i + 1, 0] - coords[i - 1, 0]) / dt
            vy[i] = (coords[i + 1, 1] - coords[i - 1, 1]) / dt

        # Backward difference for last point
        dtN = max(1e-4, timestamps[-1] - timestamps[-2])
        vx[-1] = (coords[-1, 0] - coords[-2, 0]) / dtN
        vy[-1] = (coords[-1, 1] - coords[-2, 1]) / dtN

        speed = np.hypot(vx, vy)
        speed = np.clip(speed, 0.0, max_speed_mps)

        # 2. Accelerations (ax, ay)
        ax = np.zeros(N, dtype=np.float64)
        ay = np.zeros(N, dtype=np.float64)
        accel_mag = np.zeros(N, dtype=np.float64)

        ax[0] = (vx[1] - vx[0]) / dt0
        ay[0] = (vy[1] - vy[0]) / dt0
        accel_mag[0] = (speed[1] - speed[0]) / dt0

        for i in range(1, N - 1):
            dt = max(1e-4, timestamps[i + 1] - timestamps[i - 1])
            ax[i] = (vx[i + 1] - vx[i - 1]) / dt
            ay[i] = (vy[i + 1] - vy[i - 1]) / dt
            accel_mag[i] = (speed[i + 1] - speed[i - 1]) / dt

        ax[-1] = (vx[-1] - vx[-2]) / dtN
        ay[-1] = (vy[-1] - vy[-2]) / dtN
        accel_mag[-1] = (speed[-1] - speed[-2]) / dtN

        accel_mag = np.clip(accel_mag, -max_accel_mps2, max_accel_mps2)

        # 3. Jerk (d(accel)/dt)
        jerk = np.zeros(N, dtype=np.float64)
        for i in range(1, N - 1):
            dt = max(1e-4, timestamps[i + 1] - timestamps[i - 1])
            jerk[i] = (accel_mag[i + 1] - accel_mag[i - 1]) / dt

        # 4. Heading and Angular Velocity
        headings = np.zeros(N, dtype=np.float64)
        for i in range(N):
            headings[i] = calculate_heading_deg(vx[i], vy[i])

        angular_vel = np.zeros(N, dtype=np.float64)
        for i in range(1, N - 1):
            dt = max(1e-4, timestamps[i + 1] - timestamps[i - 1])
            dh = (headings[i + 1] - headings[i - 1] + 180.0) % 360.0 - 180.0
            angular_vel[i] = dh / dt

        # Update point objects
        for i in range(N):
            points[i].velocity_x_mps = float(vx[i])
            points[i].velocity_y_mps = float(vy[i])
            points[i].speed_mps = float(speed[i])
            points[i].speed_kmh = float(speed[i] * 3.6)
            points[i].acceleration_x_mps2 = float(ax[i])
            points[i].acceleration_y_mps2 = float(ay[i])
            points[i].acceleration_magnitude_mps2 = float(accel_mag[i])
            points[i].jerk_mps3 = float(jerk[i])
            points[i].heading_deg = float(headings[i])
            points[i].angular_velocity_degps = float(angular_vel[i])

        return points
