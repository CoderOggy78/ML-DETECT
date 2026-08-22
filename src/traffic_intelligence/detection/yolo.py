"""
YOLO Object Detector backend supporting Ultralytics YOLOv8/v9/v10/v11 models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from traffic_intelligence.detection.base import BaseDetector
from traffic_intelligence.schema import Detection
from traffic_intelligence.utils.device import get_optimal_device
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.yolo")


class YOLODetector(BaseDetector):
    """Ultralytics YOLO wrapper with automatic device management and class mapping."""

    def __init__(
        self,
        model_path: Union[str, Path] = "yolov8x.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.50,
        image_size: int = 1280,
        half_precision: bool = True,
        class_mapping_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        **kwargs: Any,
    ):
        super().__init__(
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            class_mapping_path=class_mapping_path,
            device=device,
        )
        self.model_path = str(model_path)
        self.image_size = image_size
        self.half_precision = half_precision
        self.torch_device = get_optimal_device(device)
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO detector from {self.model_path} onto {self.torch_device}...")
            self.model = YOLO(self.model_path)
            # Send model to device
            if self.torch_device.type != "cpu":
                self.model.to(self.torch_device)
            logger.info(f"YOLO detector successfully initialized.")
        except Exception as e:
            logger.warning(
                f"Failed to load YOLO model from '{self.model_path}' ({e}). "
                f"Ensure ultralytics is installed and model weights are accessible."
            )
            self.model = None

    def detect(self, frame_bgr: np.ndarray, frame_id: int, timestamp_s: float) -> List[Detection]:
        if self.model is None:
            return []

        # Run inference
        results = self.model.predict(
            source=frame_bgr,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=str(self.torch_device),
            half=self.half_precision and self.torch_device.type == "cuda",
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        names = r.names

        for i in range(len(xyxy)):
            box = xyxy[i]
            conf = float(confs[i])
            cid = int(clss[i])
            raw_name = names.get(cid, str(cid))
            std_cls = self.class_mapper.map_class(raw_name)

            detections.append(
                Detection(
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=conf,
                    class_id=cid,
                    raw_class_name=raw_name,
                    standard_class=std_cls,
                )
            )

        return detections
