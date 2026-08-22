"""
Unit tests for Trajectory building, RTS Kalman smoothing, and motion kinematics estimation.
"""

import numpy as np
import pytest
from traffic_intelligence.schema import RoadUserClass, Track, TrackStateEnum, TrajectoryPoint
from traffic_intelligence.trajectories.builder import TrajectoryBuilder
from traffic_intelligence.trajectories.motion import MotionEstimator
from traffic_intelligence.trajectories.quality import TrajectoryQualityEvaluator
from traffic_intelligence.trajectories.smoothing import TrajectorySmoother


def test_trajectory_smoothing():
    # Noisy linear trajectory
    t = np.linspace(0, 10, 20)
    clean_x = 2.0 * t
    clean_y = 3.0 * t
    noisy_x = clean_x + np.random.normal(0, 0.5, size=len(t))
    noisy_y = clean_y + np.random.normal(0, 0.5, size=len(t))
    coords = np.column_stack([noisy_x, noisy_y])

    smoothed_sg = TrajectorySmoother.savitzky_golay_smooth(coords, window_length=5, polyorder=2)
    assert smoothed_sg.shape == coords.shape

    smoothed_kf = TrajectorySmoother.kalman_rts_smooth(coords, process_noise=0.1, measurement_noise=1.0)
    assert smoothed_kf.shape == coords.shape


def test_motion_estimator_kinematics():
    pts = []
    # Constant velocity: x(t) = 10*t m, y(t) = 0 -> vx = 10 m/s (36 km/h), ax = 0 m/s^2
    for i in range(10):
        t_s = i * 0.1
        pts.append(
            TrajectoryPoint(
                frame_id=i,
                timestamp_s=t_s,
                pixel_x=float(10.0 * t_s * 20.0),
                pixel_y=100.0,
                bbox_xyxy=(0, 0, 10, 10),
                world_x_m=float(10.0 * t_s),
                world_y_m=10.0,
            )
        )

    enriched = MotionEstimator.estimate_kinematics(pts)
    assert len(enriched) == 10
    # Interior points should have speed ~ 10 m/s (36 km/h)
    assert enriched[5].speed_mps == pytest.approx(10.0, abs=0.5)
    assert enriched[5].speed_kmh == pytest.approx(36.0, abs=1.8)
    assert enriched[5].acceleration_magnitude_mps2 == pytest.approx(0.0, abs=0.5)


def test_trajectory_builder(sample_transformer):
    builder = TrajectoryBuilder(transformer=sample_transformer, min_track_length=5)
    # Feed 10 observations for Track #1
    for i in range(10):
        builder.add_tracks(
            [
                Track(
                    track_id=1,
                    frame_id=i,
                    timestamp_s=i * 0.1,
                    bbox_xyxy=(float(100 + i * 5), 100.0, float(140 + i * 5), 120.0),
                    confidence=0.95,
                    class_name=RoadUserClass.CAR,
                    state=TrackStateEnum.ACTIVE,
                )
            ]
        )

    trajs = builder.build_trajectories()
    assert len(trajs) == 1
    t = trajs[0]
    assert t.track_id == 1
    assert t.class_name == RoadUserClass.CAR
    assert len(t.points) == 10
    assert t.is_valid
