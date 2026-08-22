"""
Integration test for full end-to-end pipeline execution and artifact validation.
"""

from pathlib import Path
import pytest
from traffic_intelligence.pipeline import TrafficIntelligencePipeline
from traffic_intelligence.utils.config import load_config


def test_end_to_end_synthetic_pipeline(tmp_path):
    output_dir = tmp_path / "test_pipeline_output"
    config = {
        "pipeline": {"name": "test_run", "device": "cpu"},
        "data": {"output_dir": str(output_dir), "frame_sample_rate": 1},
        "detection": {"detector_type": "dummy"},
        "tracking": {"tracker_type": "botsort"},
        "trajectories": {"min_track_length": 3},
        "discovery": {"contamination": 0.10},
    }

    pipeline = TrafficIntelligencePipeline(config)
    result = pipeline.run_on_source("synthetic")

    assert result["trajectories_count"] > 0
    assert result["quality_grade"] in {"EXCELLENT", "GOOD", "FAIR"}

    # Verify artifacts exist on disk
    assert (output_dir / "trajectories" / "trajectories.parquet").exists()
    assert (output_dir / "trajectories" / "trajectories.csv").exists()
    assert (output_dir / "tracks" / "tracks.parquet").exists()
    assert (output_dir / "events" / "events.json").exists()
    assert (output_dir / "analytics" / "traffic_summary.json").exists()
    assert (output_dir / "analytics" / "turning_movements.csv").exists()
    assert (output_dir / "analytics" / "speed_statistics.csv").exists()
    assert (output_dir / "analytics" / "conflicts.csv").exists()
    assert (output_dir / "reports" / "traffic_report.html").exists()
