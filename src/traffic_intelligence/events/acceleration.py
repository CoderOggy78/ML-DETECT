"""
Sudden Acceleration / Rapid Launch Detector.
"""

from __future__ import annotations

from typing import List
from traffic_intelligence.events.base import BaseEventDetector
from traffic_intelligence.schema import SeverityLevel, TrafficEvent, Trajectory


class SuddenAccelerationDetector(BaseEventDetector):
    """Detects aggressive or sudden positive acceleration spikes."""

    def __init__(self, hard_accel_threshold_mps2: float = 3.5):
        self.threshold = hard_accel_threshold_mps2

    def detect(self, trajectories: List[Trajectory]) -> List[TrafficEvent]:
        events: List[TrafficEvent] = []
        counter = 0

        for t in trajectories:
            for i, p in enumerate(t.points):
                a = p.acceleration_magnitude_mps2 or 0.0
                if a >= self.threshold:
                    counter += 1
                    sev = SeverityLevel.HIGH if a > 5.5 else SeverityLevel.MEDIUM
                    pos_w = (p.world_x_m, p.world_y_m) if p.world_x_m else None
                    pos_px = (p.pixel_x, p.pixel_y)

                    events.append(
                        TrafficEvent(
                            event_id=f"ACCEL_{counter:05d}",
                            event_type="SUDDEN_ACCELERATION",
                            start_timestamp_s=p.timestamp_s,
                            end_timestamp_s=p.timestamp_s + 0.5,
                            start_frame=p.frame_id,
                            end_frame=p.frame_id + 5,
                            involved_track_ids=[t.track_id],
                            primary_class=t.class_name,
                            severity=sev,
                            confidence=0.88,
                            location_world=pos_w,
                            location_pixel=pos_px,
                            description=f"Track #{t.track_id} ({t.class_name.value}) exhibited sudden acceleration of {a:.2f} m/s^2.",
                            metrics={"acceleration_mps2": a, "speed_kmh": p.speed_kmh},
                        )
                    )
                    break  # One event per trajectory instance

        return events
