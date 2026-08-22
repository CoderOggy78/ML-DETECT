"""
Pytest configuration, fixtures, and synthetic test data generators.
"""

import pytest
import numpy as np
from pathlib import Path
from traffic_intelligence.schema import RoadUserClass, MovementType, Trajectory, TrajectoryPoint, Detection
from traffic_intelligence.data.synthetic import SyntheticTrafficGenerator
from traffic_intelligence.geometry.coordinates import PixelToWorldTransformer


@pytest.fixture
def synthetic_generator():
    return SyntheticTrafficGenerator(width=1280, height=720, fps=10.0, duration_s=15.0)


@pytest.fixture
def sample_trajectories(synthetic_generator):
    trajs, _ = synthetic_generator.generate_scenario()
    return trajs


@pytest.fixture
def sample_transformer():
    return PixelToWorldTransformer(meters_per_pixel=0.05, origin_pixel=(0.0, 0.0))


@pytest.fixture
def dummy_detections():
    return [
        Detection(
            frame_id=0,
            timestamp_s=0.0,
            bbox_xyxy=(100.0, 100.0, 140.0, 120.0),
            confidence=0.95,
            class_id=2,
            raw_class_name="car",
            standard_class=RoadUserClass.CAR,
        ),
        Detection(
            frame_id=0,
            timestamp_s=0.0,
            bbox_xyxy=(200.0, 200.0, 215.0, 215.0),
            confidence=0.85,
            class_id=0,
            raw_class_name="person",
            standard_class=RoadUserClass.PEDESTRIAN,
        ),
    ]
