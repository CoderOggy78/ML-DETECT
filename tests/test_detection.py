"""
Unit tests for object detection, sliced inference, and class mapping.
"""

import numpy as np
import pytest
from traffic_intelligence.detection.base import ClassMapper, DummyDetector
from traffic_intelligence.detection.sliced import SlicedDetector, calculate_iou, non_max_merge, non_max_suppression
from traffic_intelligence.schema import Detection, RoadUserClass


def test_class_mapper():
    mapper = ClassMapper()
    assert mapper.map_class("car") == RoadUserClass.CAR
    assert mapper.map_class("Van") == RoadUserClass.LGV
    assert mapper.map_class("lorry") == RoadUserClass.HGV
    assert mapper.map_class("person") == RoadUserClass.PEDESTRIAN
    assert mapper.map_class("bicycle") == RoadUserClass.MOTORCYCLE
    assert mapper.map_class("unknown_label_xyz") == RoadUserClass.UNKNOWN


def test_iou_and_nms():
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (0.0, 0.0, 10.0, 10.0)
    box3 = (15.0, 15.0, 25.0, 25.0)

    assert calculate_iou(box1, box2) == pytest.approx(1.0)
    assert calculate_iou(box1, box3) == pytest.approx(0.0)

    dets = [
        Detection(
            frame_id=0,
            timestamp_s=0.0,
            bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
            confidence=0.9,
            class_id=0,
            raw_class_name="car",
            standard_class=RoadUserClass.CAR,
        ),
        Detection(
            frame_id=0,
            timestamp_s=0.0,
            bbox_xyxy=(1.0, 1.0, 10.0, 10.0),
            confidence=0.8,
            class_id=0,
            raw_class_name="car",
            standard_class=RoadUserClass.CAR,
        ),
    ]
    suppressed = non_max_suppression(dets, iou_threshold=0.5)
    assert len(suppressed) == 1
    assert suppressed[0].confidence == 0.9


def test_sliced_detector():
    simulated = {
        0: [
            {"bbox": [50.0, 50.0, 90.0, 90.0], "confidence": 0.95, "class_name": "car"},
            {"bbox": [700.0, 500.0, 720.0, 720.0], "confidence": 0.90, "class_name": "pedestrian"},
        ]
    }
    dummy = DummyDetector(simulated_detections=simulated)
    sliced = SlicedDetector(base_detector=dummy, slice_height=400, slice_width=400, perform_standard_pred=True)

    dummy_frame = np.zeros((800, 1000, 3), dtype=np.uint8)
    res = sliced.detect(dummy_frame, frame_id=0, timestamp_s=0.0)
    assert len(res) >= 1
