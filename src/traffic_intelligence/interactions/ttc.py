"""
Surrogate Safety Metric: Time-To-Collision (TTC) calculation between two road users.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np


def calculate_time_to_collision(
    pos_a: Tuple[float, float],
    vel_a: Tuple[float, float],
    pos_b: Tuple[float, float],
    vel_b: Tuple[float, float],
    collision_radius_m: float = 2.5,
    max_lookahead_s: float = 10.0,
) -> Optional[float]:
    """
    Computes Time-To-Collision (TTC) assuming constant velocity extrapolation:
    || (pos_a + vel_a * t) - (pos_b + vel_b * t) || <= collision_radius_m.
    
    Returns smallest positive root t in [0, max_lookahead_s], or None if trajectories do not intersect.
    """
    # Relative position and velocity: r(t) = r0 + v_rel * t
    rx = pos_a[0] - pos_b[0]
    ry = pos_a[1] - pos_b[1]
    vx = vel_a[0] - vel_b[0]
    vy = vel_a[1] - vel_b[1]

    # Quadratic equation: a*t^2 + b*t + c = R^2
    # ||r0 + v*t||^2 = (vx^2 + vy^2) t^2 + 2*(rx*vx + ry*vy) t + (rx^2 + ry^2)
    a = vx * vx + vy * vy
    b = 2.0 * (rx * vx + ry * vy)
    c = (rx * rx + ry * ry) - (collision_radius_m * collision_radius_m)

    # If already overlapping (c <= 0)
    if c <= 0:
        return 0.0

    # If relative velocity is zero or diverging (b >= 0)
    if a < 1e-6 or b >= 0:
        return None

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0:
        return None

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    # We want smallest positive time
    if t1 >= 0 and t1 <= max_lookahead_s:
        return float(t1)
    elif t2 >= 0 and t2 <= max_lookahead_s:
        return float(t2)

    return None
