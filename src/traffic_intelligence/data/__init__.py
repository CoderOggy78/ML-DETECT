"""
Data ingestion, video streaming, telemetry parsing, and synthetic dataset generation.
"""

from traffic_intelligence.data.metadata import MetadataReader
from traffic_intelligence.data.video import VideoReader, FrameIterator
from traffic_intelligence.data.telemetry import TelemetryReader
from traffic_intelligence.data.dataset import DatasetAdapter
from traffic_intelligence.data.synthetic import SyntheticTrafficGenerator

__all__ = [
    "MetadataReader",
    "VideoReader",
    "FrameIterator",
    "TelemetryReader",
    "DatasetAdapter",
    "SyntheticTrafficGenerator",
]
