"""
Unit tests for Traffic Analytics, Summary Builder, and Self-Consistency Evaluator.
"""

import pytest
from traffic_intelligence.analytics.evaluation import SelfConsistencyEvaluator
from traffic_intelligence.analytics.od import OriginDestinationAnalyzer
from traffic_intelligence.analytics.statistics import ClassWiseStatisticsCalculator
from traffic_intelligence.analytics.summary import TrafficSummaryBuilder
from traffic_intelligence.analytics.temporal import TemporalTrendAggregator


def test_class_wise_statistics(sample_trajectories):
    df = ClassWiseStatisticsCalculator.compute_summary_table(sample_trajectories)
    assert not df.empty
    assert "class_name" in df.columns
    assert "mean_speed_kmh" in df.columns


def test_self_consistency_evaluator(sample_trajectories):
    metrics = SelfConsistencyEvaluator.evaluate_unsupervised_consistency(sample_trajectories)
    assert "overall_quality_score" in metrics
    assert "data_confidence_grade" in metrics
    assert metrics["overall_quality_score"] > 0.5


def test_origin_destination_analyzer(sample_trajectories):
    for i, t in enumerate(sample_trajectories):
        t.origin_zone = "West_Approach" if i % 2 == 0 else "North_Approach"
        t.destination_zone = "East_Approach" if i % 2 == 0 else "South_Approach"

    od_pivot = OriginDestinationAnalyzer.compute_od_matrix(sample_trajectories)
    assert not od_pivot.empty
