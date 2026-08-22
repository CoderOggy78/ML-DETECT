"""
Video Overlay Renderer: draws bounding boxes, track labels, velocities, and motion trails onto video frames.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from traffic_intelligence.schema import ConflictEvent, RoadUserClass, Track, TrafficEvent


class VideoOverlayRenderer:
    """Renders professional computer vision tracking overlays and HUD onto BGR video frames."""

    CLASS_COLORS = {
        RoadUserClass.CAR: (255, 140, 0),       # Deep Sky Blue
        RoadUserClass.LGV: (200, 200, 50),      # Cyan
        RoadUserClass.HGV: (50, 150, 255),      # Orange
        RoadUserClass.BUS: (0, 215, 255),       # Gold
        RoadUserClass.TRUCK: (60, 220, 60),     # Lime Green
        RoadUserClass.MOTORCYCLE: (238, 130, 238), # Violet
        RoadUserClass.PEDESTRIAN: (0, 0, 255),  # Crimson Red
        RoadUserClass.UNKNOWN: (180, 180, 180), # Gray
    }

    def __init__(self, trail_length: int = 30):
        self.trail_length = trail_length
        self._history: Dict[int, List[Tuple[int, int]]] = {}

    def draw_frame_overlay(
        self,
        frame_bgr: np.ndarray,
        tracks: List[Track],
        conflicts: Optional[List[ConflictEvent]] = None,
        events: Optional[List[TrafficEvent]] = None,
        frame_id: int = 0,
        timestamp_s: float = 0.0,
    ) -> np.ndarray:
        """Renders bounding boxes, IDs, motion trails, and conflict markers onto frame."""
        annotated = frame_bgr.copy()

        # 1. Update and draw motion trails
        for t in tracks:
            cx, cy = int(t.center_xy[0]), int(t.center_xy[1])
            self._history.setdefault(t.track_id, []).append((cx, cy))
            if len(self._history[t.track_id]) > self.trail_length:
                self._history[t.track_id].pop(0)

            trail = self._history[t.track_id]
            color = self.CLASS_COLORS.get(t.class_name, (200, 200, 200))

            for i in range(1, len(trail)):
                thickness = int(np.clip(i / len(trail) * 3, 1, 3))
                cv2.line(annotated, trail[i - 1], trail[i], color, thickness)

        # 2. Draw Bounding Boxes and Labels
        for t in tracks:
            xmin, ymin, xmax, ymax = map(int, t.bbox_xyxy)
            color = self.CLASS_COLORS.get(t.class_name, (200, 200, 200))

            # Bounding box
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)

            # Label banner
            spd_text = f"{t.speed_mps * 3.6:.0f} km/h" if t.speed_mps is not None else ""
            label = f"#{t.track_id} {t.class_name.value} {spd_text}".strip()

            (w_lbl, h_lbl), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(
                annotated,
                (xmin, max(0, ymin - h_lbl - 6)),
                (xmin + w_lbl + 6, max(h_lbl + 6, ymin)),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (xmin + 3, max(h_lbl + 2, ymin - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # 3. Draw active conflict markers if any
        if conflicts:
            for c in conflicts:
                if c.frame_id == frame_id:
                    px, py = map(int, c.location_pixel)
                    cv2.circle(annotated, (px, py), 18, (0, 0, 255), 2)
                    cv2.circle(annotated, (px, py), 6, (0, 0, 255), -1)
                    cv2.putText(
                        annotated,
                        f"! {c.conflict_type} (TTC:{c.ttc_s:.1f}s)" if c.ttc_s else f"! {c.conflict_type}",
                        (px + 12, py),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        2,
                    )

        # 4. HUD Banner
        h, w = annotated.shape[:2]
        hud_text = f"Frame: {frame_id:05d} | Time: {timestamp_s:.2f}s | Active Targets: {len(tracks)}"
        cv2.putText(annotated, hud_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        return annotated
