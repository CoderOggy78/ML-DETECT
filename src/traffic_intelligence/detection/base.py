"""
Base Detector abstract class, class mapper, and fallback dummy detector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from traffic_intelligence.schema import Detection, RoadUserClass
from traffic_intelligence.utils.config import load_config
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.detection")


class ClassMapper:
    """Maps arbitrary external detector labels (e.g. COCO, VisDrone) to standard RoadUserClass."""

    def __init__(self, mapping_config_or_path: Optional[Union[Dict[str, Any], str, Path]] = None):
        self.mapping: Dict[str, str] = {
            "car": "CAR",
            "truck": "TRUCK",
            "bus": "BUS",
            "motorcycle": "MOTORCYCLE",
            "bicycle": "MOTORCYCLE",
            "person": "PEDESTRIAN",
            "pedestrian": "PEDESTRIAN",
            "van": "LGV",
            "minivan": "LGV",
            "pickup": "LGV",
            "lorry": "HGV",
            "trailer": "HGV",
        }
        self.default_class: str = "UNKNOWN"

        if mapping_config_or_path:
            if isinstance(mapping_config_or_path, (str, Path)):
                p = Path(mapping_config_or_path)
                if p.exists():
                    cfg = load_config(p)
                    self.mapping.update(cfg.get("mapping", {}))
                    self.default_class = cfg.get("default_class", "UNKNOWN")
            elif isinstance(mapping_config_or_path, dict):
                self.mapping.update(mapping_config_or_path.get("mapping", mapping_config_or_path))
                self.default_class = mapping_config_or_path.get("default_class", "UNKNOWN")

    def map_class(self, raw_label: Union[str, int]) -> RoadUserClass:
        clean = str(raw_label).lower().strip().replace(" ", "_")
        std_str = self.mapping.get(clean, self.default_class)
        try:
            return RoadUserClass(std_str)
        except ValueError:
            return RoadUserClass.UNKNOWN


class BaseDetector(ABC):
    """Abstract interface for all pluggable object detection models."""

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.50,
        class_mapping_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        **kwargs: Any,
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.class_mapper = ClassMapper(class_mapping_path)
        self.device = device

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray, frame_id: int, timestamp_s: float) -> List[Detection]:
        """Runs object detection on a single frame."""
        pass

    def detect_batch(
        self, frames_bgr: List[np.ndarray], frame_ids: List[int], timestamps_s: List[float]
    ) -> List[List[Detection]]:
        """Default batch inference (can be overridden with optimized batched tensors)."""
        return [self.detect(f, fid, ts) for f, fid, ts in zip(frames_bgr, frame_ids, timestamps_s)]


class DummyDetector(BaseDetector):
    """A deterministic mock detector for testing and synthetic validation without heavy GPU/weights."""

    def __init__(
        self,
        simulated_detections: Optional[Dict[int, List[Dict[str, Any]]]] = None,
        confidence_threshold: float = 0.25,
        **kwargs: Any,
    ):
        super().__init__(confidence_threshold=confidence_threshold, **kwargs)
        self.simulated_detections = simulated_detections or {}

    def detect(self, frame_bgr: np.ndarray, frame_id: int, timestamp_s: float) -> List[Detection]:
        raw_list = self.simulated_detections.get(frame_id, [])
        dets: List[Detection] = []
        for idx, item in enumerate(raw_list):
            raw_cls = item.get("class_name", "car")
            std_cls = self.class_mapper.map_class(raw_cls)
            bbox = tuple(item.get("bbox", [100.0, 100.0, 150.0, 150.0]))
            conf = float(item.get("confidence", 0.9))

            if conf >= self.confidence_threshold:
                dets.append(
                    Detection(
                        frame_id=frame_id,
                        timestamp_s=timestamp_s,
                        bbox_xyxy=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                        confidence=conf,
                        class_id=idx,
                        raw_class_name=raw_cls,
                        standard_class=std_cls,
                    )
                )
        return dets
