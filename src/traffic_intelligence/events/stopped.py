"""
Stopped Vehicle / Long Dwell Time in Active Travel Corridor Detector.
"""

from __future__ import annotations

from typing import List
from traffic_intelligence.events.base import BaseEventDetector
from traffic_intelligence.schema import SeverityLevel, TrafficEvent, Trajectory


class StoppedVehicleDetector(BaseEventDetector):
    """Detects vehicles stopped or idling in active lanes for extended durations."""

    def __init__(self, min_dwell_time_s: float = 5.0, max_speed_kmh: float = 3.0):
        self.min_dwell_time_s = min_dwell_time_s
        self.max_speed_kmh = max_speed_kmh

    def detect(self, trajectories: List[Trajectory]) -> List[TrafficEvent]:
        events: List[TrafficEvent] = []
        counter = 0

        for t in trajectories:
            if t.dwell_time_s >= self.min_dwell_time_s:
                counter += 1
                mid_pt = t.points[len(t.points) // 2]
                pos_w = (mid_pt.world_x_m, mid_pt.world_y_m) if mid_pt.world_x_m else None
                pos_px = (mid_pt.pixel_x, mid_pt.pixel_y)

                events.append(
                    TrafficEvent(
                        event_id=f"STOP_{counter:05d}",
                        event_type="STOPPED_VEHICLE",
                        start_timestamp_s=t.start_timestamp_s,
                        end_timestamp_s=t.end_timestamp_s,
                        start_frame=t.start_frame,
                        end_frame=t.end_frame,
                        involved_track_ids=[t.track_id],
                        primary_class=t.class_name,
                        severity=SeverityLevel.MEDIUM if t.dwell_time_s < 15.0 else SeverityLevel.HIGH,
                        confidence=0.95,
                        location_world=pos_w,
                        location_pixel=pos_px,
                        description=f"Track #{t.track_id} ({t.class_name.value}) was stopped/stationary for {t.dwell_time_s:.1f} seconds.",
                        metrics={"dwell_time_s": t.dwell_time_s, "average_speed_kmh": t.average_speed_kmh},
                    )
                )

        return events
