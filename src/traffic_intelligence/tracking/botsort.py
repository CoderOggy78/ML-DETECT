"""
BoT-SORT: Robust Multi-Object Tracking with Camera Motion Compensation and Appearance Re-ID Fusion.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

from traffic_intelligence.schema import Detection, RoadUserClass, Track, TrackStateEnum
from traffic_intelligence.tracking.base import BaseTracker, compute_iou_cost_matrix, linear_assignment
from traffic_intelligence.tracking.bytetrack import STrack
from traffic_intelligence.tracking.kalman import KalmanBoxTracker
from traffic_intelligence.tracking.reid import AppearanceExtractor, cosine_distance_matrix
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.botsort")


class BoTSORTTrack(STrack):
    """Extended STrack with visual feature appearance memory and exponential moving average (EMA)."""

    def __init__(
        self,
        detection: Detection,
        appearance_feature: Optional[np.ndarray] = None,
        ema_alpha: float = 0.90,
    ):
        super().__init__(detection)
        self.ema_alpha = ema_alpha
        self.appearance_feature: Optional[np.ndarray] = appearance_feature
        if appearance_feature is not None:
            norm = np.linalg.norm(appearance_feature)
            if norm > 1e-6:
                self.appearance_feature = appearance_feature / norm

    def update_appearance(self, new_feature: np.ndarray) -> None:
        norm = np.linalg.norm(new_feature)
        if norm > 1e-6:
            unit_feat = new_feature / norm
            if self.appearance_feature is None:
                self.appearance_feature = unit_feat
            else:
                self.appearance_feature = (
                    self.ema_alpha * self.appearance_feature + (1.0 - self.ema_alpha) * unit_feat
                )
                self.appearance_feature /= np.linalg.norm(self.appearance_feature)


class BoTSORTTracker(BaseTracker):
    """
    BoT-SORT implementation: Combines Camera Motion Compensation (CMC),
    appearance feature fusion, and two-stage association.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.50,
        track_low_thresh: float = 0.15,
        new_track_thresh: float = 0.60,
        match_thresh: float = 0.80,
        proximity_thresh: float = 0.50,
        appearance_thresh: float = 0.35,
        track_buffer: int = 60,
        cmc_enabled: bool = True,
        reid_enabled: bool = True,
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.track_buffer = track_buffer
        self.cmc_enabled = cmc_enabled
        self.reid_enabled = reid_enabled

        self.extractor = AppearanceExtractor() if reid_enabled else None
        self.tracked_stracks: List[BoTSORTTrack] = []
        self.lost_stracks: List[BoTSORTTrack] = []
        self.removed_stracks: List[BoTSORTTrack] = []
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
        cmc = cmc_matrix if self.cmc_enabled else None

        # Extract appearance features for detections
        det_features: Optional[np.ndarray] = None
        if self.reid_enabled and self.extractor is not None and frame_bgr is not None and len(detections) > 0:
            det_boxes = [d.bbox_xyxy for d in detections]
            det_features = self.extractor.extract_features(frame_bgr, det_boxes)

        # 1. Partition detections
        det_high: List[Tuple[Detection, Optional[np.ndarray]]] = []
        det_low: List[Tuple[Detection, Optional[np.ndarray]]] = []

        for idx, d in enumerate(detections):
            feat = det_features[idx] if det_features is not None else None
            if d.confidence >= self.track_high_thresh:
                det_high.append((d, feat))
            elif d.confidence >= self.track_low_thresh:
                det_low.append((d, feat))

        # 2. Predict Kalman Filter state for all tracks with CMC
        unconfirmed: List[BoTSORTTrack] = []
        tracked_pool: List[BoTSORTTrack] = []
        for track in self.tracked_stracks:
            if track.state == TrackStateEnum.ACTIVE:
                tracked_pool.append(track)
            else:
                unconfirmed.append(track)

        pool_for_first_match = tracked_pool + self.lost_stracks
        for t in pool_for_first_match:
            t.predict(cmc)

        activated_stracks: List[BoTSORTTrack] = []
        refind_stracks: List[BoTSORTTrack] = []
        lost_stracks: List[BoTSORTTrack] = []
        removed_stracks: List[BoTSORTTrack] = []

        # 3. First Association: Combined Appearance + IoU cost
        if pool_for_first_match and det_high:
            boxes_pool = [t.current_bbox for t in pool_for_first_match]
            boxes_high = [d[0].bbox_xyxy for d in det_high]
            iou_cost = compute_iou_cost_matrix(boxes_pool, boxes_high)

            if self.reid_enabled and any(t.appearance_feature is not None for t in pool_for_first_match):
                pool_feats = np.array(
                    [
                        t.appearance_feature
                        if t.appearance_feature is not None
                        else np.zeros(self.extractor.feature_dim)
                        for t in pool_for_first_match
                    ]
                )
                high_feats = np.array(
                    [
                        d[1]
                        if d[1] is not None
                        else np.zeros(self.extractor.feature_dim)
                        for d in det_high
                    ]
                )
                app_cost = cosine_distance_matrix(pool_feats, high_feats)
                # Combined cost: fuse IoU and appearance
                cost_matrix = np.where(
                    iou_cost < self.proximity_thresh,
                    np.minimum(iou_cost, app_cost),
                    iou_cost,
                )
            else:
                cost_matrix = iou_cost

            matches, u_tracks, u_dets_high = linear_assignment(cost_matrix, threshold=self.match_thresh)
        else:
            matches = np.empty((0, 2), dtype=int)
            u_tracks = np.arange(len(pool_for_first_match), dtype=int)
            u_dets_high = np.arange(len(det_high), dtype=int)

        for itracked, idet in matches:
            track = pool_for_first_match[itracked]
            det, feat = det_high[idet]
            track.update(det)
            if feat is not None:
                track.update_appearance(feat)

            if track.state == TrackStateEnum.ACTIVE:
                activated_stracks.append(track)
            else:
                refind_stracks.append(track)

        # 4. Second Association with low-confidence detections
        r_tracked_stracks = [pool_for_first_match[i] for i in u_tracks if pool_for_first_match[i].state == TrackStateEnum.ACTIVE]
        if r_tracked_stracks and det_low:
            boxes_r_tracked = [t.current_bbox for t in r_tracked_stracks]
            boxes_low = [d[0].bbox_xyxy for d in det_low]
            cost_matrix_low = compute_iou_cost_matrix(boxes_r_tracked, boxes_low)
            matches_low, u_r_tracks, _ = linear_assignment(cost_matrix_low, threshold=0.50)
        else:
            matches_low = np.empty((0, 2), dtype=int)
            u_r_tracks = np.arange(len(r_tracked_stracks), dtype=int)

        for itracked, idet in matches_low:
            track = r_tracked_stracks[itracked]
            det, feat = det_low[idet]
            track.update(det)
            if feat is not None:
                track.update_appearance(feat)
            activated_stracks.append(track)

        for it in u_r_tracks:
            track = r_tracked_stracks[it]
            track.mark_lost()
            lost_stracks.append(track)

        # 5. Handle unconfirmed tracks
        u_det_high_list = [det_high[i] for i in u_dets_high]
        if unconfirmed and u_det_high_list:
            boxes_unc = [t.current_bbox for t in unconfirmed]
            boxes_u_high = [d[0].bbox_xyxy for d in u_det_high_list]
            cost_unc = compute_iou_cost_matrix(boxes_unc, boxes_u_high)
            matches_unc, u_unc, u_remain_det = linear_assignment(cost_unc, threshold=0.70)
        else:
            matches_unc = np.empty((0, 2), dtype=int)
            u_unc = np.arange(len(unconfirmed), dtype=int)
            u_remain_det = np.arange(len(u_det_high_list), dtype=int)

        for itracked, idet in matches_unc:
            det, feat = u_det_high_list[idet]
            unconfirmed[itracked].update(det)
            if feat is not None:
                unconfirmed[itracked].update_appearance(feat)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unc:
            unconfirmed[it].mark_terminated()
            removed_stracks.append(unconfirmed[it])

        # 6. Initialize new tracks
        for idet in u_remain_det:
            det, feat = u_det_high_list[idet]
            if det.confidence >= self.new_track_thresh:
                new_track = BoTSORTTrack(det, appearance_feature=feat)
                activated_stracks.append(new_track)

        # 7. Lifecycle update
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.track_buffer:
                track.mark_terminated()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in activated_stracks if t.state in {TrackStateEnum.ACTIVE, TrackStateEnum.NEW}]
        self.lost_stracks = [
            t for t in (self.lost_stracks + lost_stracks) if t.state == TrackStateEnum.LOST and t not in refind_stracks
        ]
        self.removed_stracks.extend(removed_stracks)

        # Output schema conversion
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
                    feature_embedding=t.appearance_feature.tolist() if t.appearance_feature is not None else None,
                )
            )

        return output_tracks
