"""
Unit tests for Behavioral Event Detectors (Braking, Acceleration, Stopped, Wrong-Way, Cut-In).
"""

import pytest
from traffic_intelligence.events.acceleration import SuddenAccelerationDetector
from traffic_intelligence.events.braking import SuddenBrakingDetector
from traffic_intelligence.events.cut_in import CutInDetector
from traffic_intelligence.events.engine import EventEngine
from traffic_intelligence.events.stopped import StoppedVehicleDetector
from traffic_intelligence.events.wrong_way import WrongWayDetector


def test_sudden_braking_detector(sample_trajectories):
    detector = SuddenBrakingDetector(hard_braking_threshold_mps2=-3.0)
    events = detector.detect(sample_trajectories)
    assert isinstance(events, list)
    # Synthetic scenario agent 2 has sudden braking
    brake_events = [e for e in events if e.event_type == "SUDDEN_BRAKING"]
    assert len(brake_events) >= 1


def test_stopped_vehicle_detector(sample_trajectories):
    detector = StoppedVehicleDetector(min_dwell_time_s=1.0)
    events = detector.detect(sample_trajectories)
    # Synthetic scenario agent 7 is stopped
    stop_events = [e for e in events if e.event_type == "STOPPED_VEHICLE"]
    assert len(stop_events) >= 1


def test_wrong_way_detector(sample_trajectories):
    detector = WrongWayDetector(wrong_way_angle_threshold_deg=100.0)
    events = detector.detect(sample_trajectories)
    # Agent 8 moves leftward counter to eastbound flow
    wrong_events = [e for e in events if e.event_type == "WRONG_WAY"]
    assert len(wrong_events) >= 1


def test_event_engine(sample_trajectories):
    engine = EventEngine()
    all_events = engine.detect_all(sample_trajectories)
    assert len(all_events) > 0
