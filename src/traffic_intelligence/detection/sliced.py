"""
Sliced / Tiled Inference Engine (SAHI pattern) for ultra-small road user detection in aerial views.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np

from traffic_intelligence.detection.base import BaseDetector
from traffic_intelligence.schema import Detection, RoadUserClass
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.sliced")


def calculate_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes (xmin, ymin, xmax, ymax)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def non_max_suppression(detections: List[Detection], iou_threshold: float = 0.50) -> List[Detection]:
    """Standard Non-Maximum Suppression (NMS) across detections."""
    if not detections:
        return []

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep: List[Detection] = []

    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)
        sorted_dets = [
            d for d in sorted_dets
            if d.standard_class != best.standard_class or calculate_iou(d.bbox_xyxy, best.bbox_xyxy) < iou_threshold
        ]

    return keep


def non_max_merge(detections: List[Detection], match_threshold: float = 0.50) -> List[Detection]:
    """
    Non-Maximum Merging (NMM): clusters overlapping detections and computes confidence-weighted average box.
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    merged: List[Detection] = []
    visited = [False] * len(sorted_dets)

    for i in range(len(sorted_dets)):
        if visited[i]:
            continue

        cluster = [sorted_dets[i]]
        visited[i] = True

        for j in range(i + 1, len(sorted_dets)):
            if visited[j]:
                continue
            if sorted_dets[i].standard_class == sorted_dets[j].standard_class:
                if calculate_iou(sorted_dets[i].bbox_xyxy, sorted_dets[j].bbox_xyxy) >= match_threshold:
                    cluster.append(sorted_dets[j])
                    visited[j] = True

        # Compute weighted box
        total_conf = sum(d.confidence for d in cluster)
        avg_xmin = sum(d.bbox_xyxy[0] * d.confidence for d in cluster) / total_conf
        avg_ymin = sum(d.bbox_xyxy[1] * d.confidence for d in cluster) / total_conf
        avg_xmax = sum(d.bbox_xyxy[2] * d.confidence for d in cluster) / total_conf
        avg_ymax = sum(d.bbox_xyxy[3] * d.confidence for d in cluster) / total_conf
        max_conf = max(d.confidence for d in cluster)

        merged.append(
            Detection(
                frame_id=cluster[0].frame_id,
                timestamp_s=cluster[0].timestamp_s,
                bbox_xyxy=(float(avg_xmin), float(avg_ymin), float(avg_xmax), float(avg_ymax)),
                confidence=max_conf,
                class_id=cluster[0].class_id,
                raw_class_name=cluster[0].raw_class_name,
                standard_class=cluster[0].standard_class,
            )
        )

    return merged


class SlicedDetector(BaseDetector):
    """
    Wraps any BaseDetector to perform sliding-window tiled sliced inference
    and multi-scale merging for high-altitude aerial imagery.
    """

    def __init__(
        self,
        base_detector: BaseDetector,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.20,
        overlap_width_ratio: float = 0.20,
        perform_standard_pred: bool = True,
        merge_type: str = "NMM",
        merge_iou_threshold: float = 0.50,
    ):
        super().__init__(
            confidence_threshold=base_detector.confidence_threshold,
            iou_threshold=base_detector.iou_threshold,
        )
        self.base_detector = base_detector
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.perform_standard_pred = perform_standard_pred
        self.merge_type = merge_type
        self.merge_iou_threshold = merge_iou_threshold

    def _generate_slices(self, image_height: int, image_width: int) -> List[Tuple[int, int, int, int]]:
        """Computes slice bounding boxes (xmin, ymin, xmax, ymax) covering the frame with overlap."""
        step_x = int(self.slice_width * (1.0 - self.overlap_width_ratio))
        step_y = int(self.slice_height * (1.0 - self.overlap_height_ratio))

        slices = []
        y = 0
        while y < image_height:
            x = 0
            while x < image_width:
                x_end = min(x + self.slice_width, image_width)
                y_end = min(y + self.slice_height, image_height)
                x_start = max(0, x_end - self.slice_width)
                y_start = max(0, y_end - self.slice_height)

                slices.append((x_start, y_start, x_end, y_end))
                if x_end == image_width:
                    break
                x += step_x
            if y_end == image_height:
                break
            y += step_y

        return list(set(slices))

    def detect(self, frame_bgr: np.ndarray, frame_id: int, timestamp_s: float) -> List[Detection]:
        h, w = frame_bgr.shape[:2]
        all_detections: List[Detection] = []

        # 1. Full-frame standard prediction if enabled
        if self.perform_standard_pred:
            full_dets = self.base_detector.detect(frame_bgr, frame_id, timestamp_s)
            all_detections.extend(full_dets)

        # 2. Sliced predictions
        slices = self._generate_slices(h, w)
        for x_start, y_start, x_end, y_end in slices:
            crop = frame_bgr[y_start:y_end, x_start:x_end]
            if crop.size == 0:
                continue

            crop_dets = self.base_detector.detect(crop, frame_id, timestamp_s)
            # Map coordinates back to full image frame
            for d in crop_dets:
                c_xmin, c_ymin, c_xmax, c_ymax = d.bbox_xyxy
                full_box = (
                    c_xmin + x_start,
                    c_ymin + y_start,
                    c_xmax + x_start,
                    c_ymax + y_start,
                )
                all_detections.append(
                    Detection(
                        frame_id=frame_id,
                        timestamp_s=timestamp_s,
                        bbox_xyxy=full_box,
                        confidence=d.confidence,
                        class_id=d.class_id,
                        raw_class_name=d.raw_class_name,
                        standard_class=d.standard_class,
                    )
                )

        # 3. Postprocess merge
        if self.merge_type.upper() == "NMM":
            return non_max_merge(all_detections, match_threshold=self.merge_iou_threshold)
        else:
            return non_max_suppression(all_detections, iou_threshold=self.merge_iou_threshold)
