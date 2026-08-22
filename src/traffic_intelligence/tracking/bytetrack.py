"""
ByteTrack: Multi-Object Tracking by Associating Every Detection Box (Zhang et al.).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from traffic_intelligence.schema import Detection, RoadUserClass, Track, TrackStateEnum
from traffic_intelligence.tracking.base import BaseTracker, compute_iou_cost_matrix, linear_assignment
from traffic_intelligence.tracking.kalman import KalmanBoxTracker
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.bytetrack")


class STrack:
    """Individual Single Object Track container."""

    def __init__(
        self,
        detection: Detection,
        std_weight_position: float = 0.05,
        std_weight_velocity: float = 0.00625,
    ):
        self.kalman = KalmanBoxTracker(
            detection.bbox_xyxy,
            std_weight_position=std_weight_position,
            std_weight_velocity=std_weight_velocity,
        )
        self.track_id = self.kalman.id
        self.class_name: RoadUserClass = detection.standard_class
        self.confidence: float = detection.confidence
        self.state: TrackStateEnum = TrackStateEnum.NEW
        self.start_frame: int = detection.frame_id
        self.frame_id: int = detection.frame_id
        self.timestamp_s: float = detection.timestamp_s
        self.hit_count: int = 1
        self.lost_frames: int = 0
        self.history_bboxes: List[Tuple[float, float, float, float]] = [detection.bbox_xyxy]

    def predict(self, cmc_matrix: Optional[np.ndarray] = None) -> Tuple[float, float, float, float]:
        box = self.kalman.predict(cmc_matrix)
        return box

    def update(self, detection: Detection) -> None:
        self.kalman.update(detection.bbox_xyxy)
        self.frame_id = detection.frame_id
        self.timestamp_s = detection.timestamp_s
        self.confidence = detection.confidence
        self.class_name = detection.standard_class
        self.hit_count += 1
        self.lost_frames = 0
        self.state = TrackStateEnum.ACTIVE
        self.history_bboxes.append(detection.bbox_xyxy)

    def mark_lost(self) -> None:
        self.state = TrackStateEnum.LOST
        self.lost_frames += 1

    def mark_terminated(self) -> None:
        self.state = TrackStateEnum.TERMINATED

    @property
    def current_bbox(self) -> Tuple[float, float, float, float]:
        return self.kalman.get_state_bbox()


class ByteTracker(BaseTracker):
    """
    ByteTrack implementation that associates high-confidence detections first,
    then pairs remaining lost tracks with low-confidence detections.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.50,
        track_low_thresh: float = 0.15,
        new_track_thresh: float = 0.60,
        match_thresh: float = 0.80,
        track_buffer: int = 60,
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.frame_id: int = 0

    def reset(self) -> None:
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
        KalmanBoxTracker.count = 0

    def update(
        self,
        detections: List[Detection],
        frame_bgr: Optional[np.ndarray] = None,
        cmc_matrix: Optional[np.ndarray] = None,
    ) -> List[Track]:
        self.frame_id += 1
        activated_stracks: List[STrack] = []
        refind_stracks: List[STrack] = []
        lost_stracks: List[STrack] = []
        removed_stracks: List[STrack] = []

        # 1. Split detections into high and low confidence sets
        det_high: List[Detection] = []
        det_low: List[Detection] = []
        for d in detections:
            if d.confidence >= self.track_high_thresh:
                det_high.append(d)
            elif d.confidence >= self.track_low_thresh:
                det_low.append(d)

        # 2. Predict all track states using Kalman filter with CMC
        unconfirmed: List[STrack] = []
        tracked_pool: List[STrack] = []
        for track in self.tracked_stracks:
            if track.state == TrackStateEnum.ACTIVE:
                tracked_pool.append(track)
            else:
                unconfirmed.append(track)

        pool_for_first_match = tracked_pool + self.lost_stracks
        for t in pool_for_first_match:
            t.predict(cmc_matrix)

        # 3. First Association: match high-confidence detections with active & lost tracks
        boxes_pool = [t.current_bbox for t in pool_for_first_match]
        boxes_high = [d.bbox_xyxy for d in det_high]
        cost_matrix = compute_iou_cost_matrix(boxes_pool, boxes_high)

        matches, u_tracks, u_dets_high = linear_assignment(cost_matrix, threshold=self.match_thresh)

        for itracked, idet in matches:
            track = pool_for_first_match[itracked]
            det = det_high[idet]
            if track.state == TrackStateEnum.ACTIVE:
                track.update(det)
                activated_stracks.append(track)
            else:
                track.update(det)
                refind_stracks.append(track)

        # 4. Second Association: match remaining active tracks with low-confidence detections
        r_tracked_stracks = [pool_for_first_match[i] for i in u_tracks if pool_for_first_match[i].state == TrackStateEnum.ACTIVE]
        boxes_r_tracked = [t.current_bbox for t in r_tracked_stracks]
        boxes_low = [d.bbox_xyxy for d in det_low]
        cost_matrix_low = compute_iou_cost_matrix(boxes_r_tracked, boxes_low)

        matches_low, u_r_tracks, _ = linear_assignment(cost_matrix_low, threshold=0.50)

        for itracked, idet in matches_low:
            track = r_tracked_stracks[itracked]
            det = det_low[idet]
            track.update(det)
            activated_stracks.append(track)

        for it in u_r_tracks:
            track = r_tracked_stracks[it]
            track.mark_lost()
            lost_stracks.append(track)

        # 5. Deal with unconfirmed tracks
        u_det_high_list = [det_high[i] for i in u_dets_high]
        boxes_unconfirmed = [t.current_bbox for t in unconfirmed]
        boxes_u_high = [d.bbox_xyxy for d in u_det_high_list]
        cost_unconfirmed = compute_iou_cost_matrix(boxes_unconfirmed, boxes_u_high)

        matches_unc, u_unc, u_remain_det = linear_assignment(cost_unconfirmed, threshold=0.70)

        for itracked, idet in matches_unc:
            unconfirmed[itracked].update(u_det_high_list[idet])
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unc:
            unconfirmed[it].mark_terminated()
            removed_stracks.append(unconfirmed[it])

        # 6. Initialize new tracks from unmatched high-confidence detections
        for idet in u_remain_det:
            det = u_det_high_list[idet]
            if det.confidence >= self.new_track_thresh:
                new_track = STrack(det)
                activated_stracks.append(new_track)

        # 7. Update lost and removed tracks
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.track_buffer:
                track.mark_terminated()
                removed_stracks.append(track)

        # Retain tracked lists
        self.tracked_stracks = [t for t in activated_stracks if t.state in {TrackStateEnum.ACTIVE, TrackStateEnum.NEW}]
        self.lost_stracks = [
            t for t in (self.lost_stracks + lost_stracks) if t.state == TrackStateEnum.LOST and t not in refind_stracks
        ]
        self.removed_stracks.extend(removed_stracks)

        # Convert active tracks to standard output schema
        output_tracks: List[Track] = []
        for t in self.tracked_stracks + refind_stracks:
            box = t.current_bbox
            vx, vy = t.kalman.get_velocity_xy()
            speed = float(np.hypot(vx, vy))
            heading = float((np.degrees(np.arctan2(vy, vx)) + 360.0) % 360.0)

            output_tracks.append(
                Track(
                    track_id=t.track_id,
                    frame_id=t.frame_id,
                    timestamp_s=t.timestamp_s,
                    bbox_xyxy=box,
                    confidence=t.confidence,
                    class_name=t.class_name,
                    state=t.state,
                    velocity_x_mps=vx,
                    velocity_y_mps=vy,
                    speed_mps=speed,
                    heading_deg=heading,
                )
            )

        return output_tracks
