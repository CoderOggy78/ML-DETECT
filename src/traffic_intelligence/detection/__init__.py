"""
Pluggable object detection interfaces, YOLO/RT-DETR implementations, and Sliced/Tiled small object inference.
"""

from traffic_intelligence.detection.base import BaseDetector, DummyDetector
from traffic_intelligence.detection.yolo import YOLODetector
from traffic_intelligence.detection.sliced import SlicedDetector, non_max_suppression, non_max_merge
from traffic_intelligence.detection.registry import DetectorRegistry

__all__ = [
    "BaseDetector",
    "DummyDetector",
    "YOLODetector",
    "SlicedDetector",
    "non_max_suppression",
    "non_max_merge",
    "DetectorRegistry",
]
