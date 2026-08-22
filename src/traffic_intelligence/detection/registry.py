"""
Detector Registry and Factory for dynamic instantiation from configuration.
"""

from __future__ import annotations

from typing import Any, Dict, Type
from traffic_intelligence.detection.base import BaseDetector, DummyDetector
from traffic_intelligence.detection.yolo import YOLODetector
from traffic_intelligence.detection.sliced import SlicedDetector
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.registry")


class DetectorRegistry:
    """Factory registry for creating detector instances."""

    _registry: Dict[str, Type[BaseDetector]] = {
        "yolo": YOLODetector,
        "yolov8": YOLODetector,
        "yolov9": YOLODetector,
        "yolov10": YOLODetector,
        "yolov11": YOLODetector,
        "dummy": DummyDetector,
        "mock": DummyDetector,
    }

    @classmethod
    def register(cls, name: str, detector_cls: Type[BaseDetector]) -> None:
        cls._registry[name.lower()] = detector_cls

    @classmethod
    def create(cls, config: Dict[str, Any]) -> BaseDetector:
        det_cfg = config.get("detection", config)
        detector_type = str(det_cfg.get("detector_type", det_cfg.get("backend", "yolo"))).lower()

        detector_cls = cls._registry.get(detector_type, YOLODetector)
        logger.info(f"Instantiating detector backend: '{detector_type}' ({detector_cls.__name__})")

        # Base detector kwargs
        kwargs = {
            "model_path": det_cfg.get("model_path", det_cfg.get("model_name", "yolov8x.pt")),
            "confidence_threshold": det_cfg.get("confidence_threshold", 0.25),
            "iou_threshold": det_cfg.get("iou_threshold", 0.50),
            "image_size": det_cfg.get("image_size", 1280),
            "class_mapping_path": det_cfg.get("class_mapping_path"),
            "device": config.get("pipeline", {}).get("device", "auto"),
        }

        base_instance = detector_cls(**kwargs)

        # Check if sliced inference is enabled
        sliced_cfg = det_cfg.get("sliced_inference", {})
        if sliced_cfg.get("enabled", False):
            logger.info("Enabling Sliced/Tiled Aerial Inference wrapper.")
            return SlicedDetector(
                base_detector=base_instance,
                slice_height=sliced_cfg.get("slice_height", 640),
                slice_width=sliced_cfg.get("slice_width", 640),
                overlap_height_ratio=sliced_cfg.get("overlap_height_ratio", 0.20),
                overlap_width_ratio=sliced_cfg.get("overlap_width_ratio", 0.20),
                perform_standard_pred=sliced_cfg.get("perform_standard_pred", True),
                merge_type=sliced_cfg.get("merge_type", "NMM"),
                merge_iou_threshold=sliced_cfg.get("merge_iou_threshold", 0.50),
            )

        return base_instance
