"""
Unit tests for Road Network, Turning Movement Classification, Queues, Congestion, and Flow.
"""

import pytest
from traffic_intelligence.schema import MovementType
from traffic_intelligence.traffic.congestion import CongestionAnalyzer
from traffic_intelligence.traffic.flow import TrafficFlowEstimator
from traffic_intelligence.traffic.movements import MovementClassifier
from traffic_intelligence.traffic.queues import QueueDetector
from traffic_intelligence.traffic.road_model import RoadModel


def test_movement_classification(sample_trajectories):
    # Classify movements for synthetic scenario
    road_model = RoadModel.auto_discover_from_trajectories(sample_trajectories)
    trajs = MovementClassifier.process_all_trajectories(sample_trajectories, road_model)

    movements = {t.track_id: t.movement_type for t in trajs}
    # Agent 1 (straight)
    assert movements[1] in {MovementType.STRAIGHT, MovementType.ENTRY, MovementType.EXIT}
    # Agent 4 (turn)
    assert movements[4] in {MovementType.LEFT_TURN, MovementType.RIGHT_TURN}
    # Agent 7 (stopped)
    assert movements[7] == MovementType.STOPPED


def test_queue_detection(sample_trajectories):
    detector = QueueDetector(speed_threshold_kmh=10.0, max_vehicle_gap_m=20.0, min_queue_vehicles=1)
    df = detector.process_all_trajectories(sample_trajectories)
    assert not df.empty


def test_congestion_and_flow(sample_trajectories):
    analyzer = CongestionAnalyzer()
    res = analyzer.analyze_trajectories(sample_trajectories)
    assert "overall_mean_speed_kmh" in res
    assert "congestion_level" in res

    flow_df = TrafficFlowEstimator.compute_binned_flow(sample_trajectories, bin_size_s=1.0)
    assert not flow_df.empty
