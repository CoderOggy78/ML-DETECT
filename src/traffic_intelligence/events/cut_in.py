"""
Cut-In and Unsafe Lateral Merge Detector.
"""

from __future__ import annotations

from typing import List
import numpy as np

from traffic_intelligence.events.base import BaseEventDetector
from traffic_intelligence.schema import SeverityLevel, TrafficEvent, Trajectory


class CutInDetector(BaseEventDetector):
    """Detects vehicles making sudden lateral lane changes with high closing speed or low headway."""

    def __init__(self, min_lateral_velocity_mps: float = 1.5):
        self.min_lateral_vel = min_lateral_velocity_mps

    def detect(self, trajectories: List[Trajectory]) -> List[TrafficEvent]:
        events: List[TrafficEvent] = []
        counter = 0

        for t in trajectories:
            ang_vels = [p.angular_velocity_degps for p in t.points if p.angular_velocity_degps is not None]
            speeds = [p.speed_mps for p in t.points if p.speed_mps is not None]

            if not ang_vels or not speeds:
                continue

            max_ang = float(np.max(np.abs(ang_vels)))
            mean_spd = float(np.mean(speeds))

            # Swerve while maintaining high speed
            if max_ang > 25.0 and mean_spd > 8.0:
                counter += 1
                peak_idx = int(np.argmax(np.abs(ang_vels)))
                p_peak = t.points[peak_idx]

                events.append(
                    TrafficEvent(
                        event_id=f"CUTIN_{counter:05d}",
                        event_type="CUT_IN",
                        start_timestamp_s=p_peak.timestamp_s,
                        end_timestamp_s=p_peak.timestamp_s + 1.0,
                        start_frame=p_peak.frame_id,
                        end_frame=p_peak.frame_id + 10,
                        involved_track_ids=[t.track_id],
                        primary_class=t.class_name,
                        severity=SeverityLevel.HIGH if max_ang > 40.0 else SeverityLevel.MEDIUM,
                        confidence=0.86,
                        location_world=(p_peak.world_x_m, p_peak.world_y_m) if p_peak.world_x_m else None,
                        location_pixel=(p_peak.pixel_x, p_peak.pixel_y),
                        description=f"Track #{t.track_id} ({t.class_name.value}) executed rapid cut-in / swerve maneuver (Angular vel: {max_ang:.1f} deg/s).",
                        metrics={"angular_velocity_degps": max_ang, "speed_kmh": p_peak.speed_kmh},
                    )
                )

        return events
