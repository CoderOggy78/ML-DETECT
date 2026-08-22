"""
Surrogate Safety Metric: Deceleration Rate to Avoid Collision (DRAC).
DRAC = (Delta_v)^2 / (2 * Delta_d) where Delta_v is closing speed and Delta_d is current clearance distance.
"""

from __future__ import annotations

from typing import Optional, Tuple


def calculate_deceleration_rate_to_avoid_collision(
    distance_m: float, closing_speed_mps: float, buffer_distance_m: float = 2.0
) -> Optional[float]:
    """
    Computes required braking deceleration (m/s^2) for the following/conflicting vehicle
    to avoid collision given distance and closing speed.
    """
    if closing_speed_mps <= 0.0:
        return 0.0

    clearance = max(0.1, distance_m - buffer_distance_m)
    drac = (closing_speed_mps ** 2) / (2.0 * clearance)
    return float(drac)
