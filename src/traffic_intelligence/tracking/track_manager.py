"""
Track Lifecycle Manager and Long-Term Re-Identification Buffer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from traffic_intelligence.schema import Detection, Track, TrackStateEnum
from traffic_intelligence.tracking.base import BaseTracker
from traffic_intelligence.tracking.botsort import BoTSORTTracker
from traffic_intelligence.tracking.bytetrack import ByteTracker
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.track_manager")


class TrackManager:
    """Orchestrates tracker selection, long-term identity maintenance, and track lifecycle."""

    def __init__(self, config: Dict[str, Any]):
        trk_cfg = config.get("tracking", config)
        tracker_type = str(trk_cfg.get("tracker_type", "botsort")).lower()

        if tracker_type == "bytetrack":
            logger.info("Initializing ByteTrack tracker backend.")
            self.tracker: BaseTracker = ByteTracker(
                track_high_thresh=trk_cfg.get("track_high_thresh", 0.50),
                track_low_thresh=trk_cfg.get("track_low_thresh", 0.15),
                new_track_thresh=trk_cfg.get("new_track_thresh", 0.60),
                match_thresh=trk_cfg.get("match_thresh", 0.80),
                track_buffer=trk_cfg.get("track_buffer", 60),
            )
        else:
            logger.info("Initializing BoT-SORT tracker backend with CMC and Re-ID.")
            self.tracker = BoTSORTTracker(
                track_high_thresh=trk_cfg.get("track_high_thresh", 0.50),
                track_low_thresh=trk_cfg.get("track_low_thresh", 0.15),
                new_track_thresh=trk_cfg.get("new_track_thresh", 0.60),
                match_thresh=trk_cfg.get("match_thresh", 0.80),
                proximity_thresh=trk_cfg.get("proximity_thresh", 0.50),
                appearance_thresh=trk_cfg.get("appearance_thresh", 0.35),
                track_buffer=trk_cfg.get("track_buffer", 60),
                cmc_enabled=trk_cfg.get("cmc_enabled", True),
                reid_enabled=trk_cfg.get("reid_enabled", True),
            )

        self.long_term_buffer: Dict[int, Track] = {}
        self.long_term_retention_frames = trk_cfg.get("long_term_retention_frames", 150)

    def reset(self) -> None:
        self.tracker.reset()
        self.long_term_buffer.clear()

    def update(
        self,
        detections: List[Detection],
        frame_bgr: Optional[np.ndarray] = None,
        cmc_matrix: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """Runs multi-object tracking for the frame."""
        tracks = self.tracker.update(detections, frame_bgr=frame_bgr, cmc_matrix=cmc_matrix)

        # Update long-term retention buffer
        for t in tracks:
            self.long_term_buffer[t.track_id] = t

        return tracks
