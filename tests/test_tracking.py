"""
Unit tests for multi-object tracking (ByteTrack, BoT-SORT), Kalman filter, and Re-ID.
"""

import numpy as np
import pytest
from traffic_intelligence.schema import Detection, RoadUserClass
from traffic_intelligence.tracking.botsort import BoTSORTTracker
from traffic_intelligence.tracking.bytetrack import ByteTracker
from traffic_intelligence.tracking.kalman import KalmanBoxTracker
from traffic_intelligence.tracking.reid import AppearanceExtractor, cosine_distance_matrix


def test_kalman_box_tracker():
    init_box = (100.0, 100.0, 150.0, 150.0)
    kf = KalmanBoxTracker(init_box)

    pred_box = kf.predict()
    assert len(pred_box) == 4
    assert pred_box[2] > pred_box[0]
    assert pred_box[3] > pred_box[1]

    # Update with new measurement
    meas_box = (105.0, 102.0, 155.0, 152.0)
    kf.update(meas_box)
    assert kf.time_since_update == 0
    assert kf.hit_streak == 2


def test_appearance_extractor_and_cosine_distance():
    extractor = AppearanceExtractor(feature_dim=128)
    frame = np.full((200, 200, 3), 200, dtype=np.uint8)
    boxes = [(20.0, 20.0, 60.0, 60.0), (80.0, 80.0, 120.0, 120.0)]

    feats = extractor.extract_features(frame, boxes)
    assert feats.shape == (2, 128)

    dist_mat = cosine_distance_matrix(feats, feats)
    assert dist_mat.shape == (2, 2)
    # Diagonal should be ~0.0
    assert dist_mat[0, 0] == pytest.approx(0.0, abs=1e-3)
    assert dist_mat[1, 1] == pytest.approx(0.0, abs=1e-3)


def test_bytetracker_association(dummy_detections):
    tracker = ByteTracker()
    tracks_f0 = tracker.update(dummy_detections)
    assert len(tracks_f0) == 2

    # Advance slightly
    dets_f1 = [
        Detection(
            frame_id=1,
            timestamp_s=0.033,
            bbox_xyxy=(102.0, 101.0, 142.0, 121.0),
            confidence=0.94,
            class_id=2,
            raw_class_name="car",
            standard_class=RoadUserClass.CAR,
        ),
        Detection(
            frame_id=1,
            timestamp_s=0.033,
            bbox_xyxy=(201.0, 201.0, 216.0, 216.0),
            confidence=0.88,
            class_id=0,
            raw_class_name="person",
            standard_class=RoadUserClass.PEDESTRIAN,
        ),
    ]
    tracks_f1 = tracker.update(dets_f1)
    assert len(tracks_f1) == 2
    # IDs should be maintained
    assert tracks_f1[0].track_id == tracks_f0[0].track_id
    assert tracks_f1[1].track_id == tracks_f0[1].track_id


def test_botsort_tracker(dummy_detections):
    tracker = BoTSORTTracker(cmc_enabled=True, reid_enabled=True)
    frame = np.full((300, 300, 3), 100, dtype=np.uint8)
    tracks = tracker.update(dummy_detections, frame_bgr=frame)
    assert len(tracks) == 2
