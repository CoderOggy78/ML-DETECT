"""
Unit tests for Surrogate Safety metrics: TTC, PET, DRAC, and ConflictDetector.
"""

import pytest
from traffic_intelligence.interactions.conflicts import ConflictDetector
from traffic_intelligence.interactions.drac import calculate_deceleration_rate_to_avoid_collision
from traffic_intelligence.interactions.graph import InteractionGraph
from traffic_intelligence.interactions.pet import PostEncroachmentTimeCalculator
from traffic_intelligence.interactions.ttc import calculate_time_to_collision
from traffic_intelligence.schema import RoadUserClass, TrajectoryPoint


def test_time_to_collision():
    # Head-on collision: Agent A at x=0 moving +10 m/s, Agent B at x=100 moving -10 m/s
    # Relative speed = 20 m/s. Meeting point = x=50 at t = 100/20 = 5.0 s (with collision radius 2.0m: t ~ 4.9s)
    pos_a = (0.0, 50.0)
    vel_a = (10.0, 0.0)
    pos_b = (100.0, 50.0)
    vel_b = (-10.0, 0.0)

    ttc = calculate_time_to_collision(pos_a, vel_a, pos_b, vel_b, collision_radius_m=2.0)
    assert ttc is not None
    assert ttc == pytest.approx(4.9, abs=0.2)

    # Diverging paths (moving away from each other)
    vel_a_away = (-10.0, 0.0)
    vel_b_away = (10.0, 0.0)
    ttc_none = calculate_time_to_collision(pos_a, vel_a_away, pos_b, vel_b_away)
    assert ttc_none is None


def test_deceleration_rate_to_avoid_collision():
    # Distance = 22m, Buffer = 2m -> Clearance = 20m. Closing speed = 10 m/s
    # DRAC = 10^2 / (2 * 20) = 100 / 40 = 2.5 m/s^2
    drac = calculate_deceleration_rate_to_avoid_collision(distance_m=22.0, closing_speed_mps=10.0, buffer_distance_m=2.0)
    assert drac == pytest.approx(2.5, abs=1e-3)


def test_conflict_detector_synthetic(sample_trajectories):
    detector = ConflictDetector()
    conflicts = detector.detect_conflicts(sample_trajectories)
    assert isinstance(conflicts, list)
    # The synthetic scenario contains rear-end braking and pedestrian crossings
    assert len(conflicts) > 0
