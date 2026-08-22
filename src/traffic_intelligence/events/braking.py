"""
Sudden Braking / Emergency Deceleration Detector.
"""

from __future__ import annotations

from typing import List
import numpy as np

from traffic_intelligence.events.base import BaseEventDetector
from traffic_intelligence.schema import SeverityLevel, TrafficEvent, Trajectory
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.events.braking")


class SuddenBrakingDetector(BaseEventDetector):
    """Detects abrupt negative acceleration events below critical deceleration threshold."""

    def __init__(self, hard_braking_threshold_mps2: float = -4.5):
        self.threshold = hard_braking_threshold_mps2

    def detect(self, trajectories: List[Trajectory]) -> List[TrafficEvent]:
        events: List[TrafficEvent] = []
        counter = 0

        for t in trajectories:
            accels = [p.acceleration_magnitude_mps2 for p in t.points if p.acceleration_magnitude_mps2 is not None]
            if not accels:
                continue

            # Identify consecutive points exceeding threshold
            in_event = False
            start_idx = 0

            for i, p in enumerate(t.points):
                a = p.acceleration_magnitude_mps2 or 0.0
                if a <= self.threshold:
                    if not in_event:
                        in_event = True
                        start_idx = i
                else:
                    if in_event:
                        in_event = False
                        end_idx = i
                        if end_idx - start_idx >= 2:  # Minimum temporal persistence
                            counter += 1
                            pt_start = t.points[start_idx]
                            pt_end = t.points[end_idx]
                            min_a = float(min(p.acceleration_magnitude_mps2 or 0.0 for p in t.points[start_idx:end_idx]))

                            sev = SeverityLevel.CRITICAL if min_a < -7.0 else (
                                SeverityLevel.HIGH if min_a < -5.5 else SeverityLevel.MEDIUM
                            )

                            pos_w = (pt_start.world_x_m, pt_start.world_y_m) if pt_start.world_x_m else None
                            pos_px = (pt_start.pixel_x, pt_start.pixel_y)

                            events.append(
                                TrafficEvent(
                                    event_id=f"BRAKE_{counter:05d}",
                                    event_type="SUDDEN_BRAKING",
                                    start_timestamp_s=pt_start.timestamp_s,
                                    end_timestamp_s=pt_end.timestamp_s,
                                    start_frame=pt_start.frame_id,
                                    end_frame=pt_end.frame_id,
                                    involved_track_ids=[t.track_id],
                                    primary_class=t.class_name,
                                    severity=sev,
                                    confidence=0.92,
                                    location_world=pos_w,
                                    location_pixel=pos_px,
                                    description=f"Track #{t.track_id} ({t.class_name.value}) executed sudden braking with max deceleration of {min_a:.2f} m/s^2.",
                                    metrics={"max_deceleration_mps2": min_a, "initial_speed_kmh": pt_start.speed_kmh},
                                )
                            )

        return events
