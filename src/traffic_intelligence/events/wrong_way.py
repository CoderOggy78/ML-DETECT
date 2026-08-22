"""
Wrong-Way Movement Detector: identifies road users traveling opposite to the prevailing corridor flow.
"""

from __future__ import annotations

import math
from typing import List
import numpy as np

from traffic_intelligence.events.base import BaseEventDetector
from traffic_intelligence.geometry.coordinates import calculate_angle_between_headings
from traffic_intelligence.schema import RoadUserClass, SeverityLevel, TrafficEvent, Trajectory


class WrongWayDetector(BaseEventDetector):
    """Detects vehicles traveling counter to dominant spatial corridor traffic flows."""

    def __init__(self, wrong_way_angle_threshold_deg: float = 120.0):
        self.angle_threshold = wrong_way_angle_threshold_deg

    def detect(self, trajectories: List[Trajectory]) -> List[TrafficEvent]:
        events: List[TrafficEvent] = []
        if len(trajectories) < 2:
            return events

        # Exclude pedestrians and stopped vehicles from flow heading baseline
        moving_vehicles = [
            t for t in trajectories
            if t.class_name != RoadUserClass.PEDESTRIAN and t.average_speed_kmh > 10.0
        ]
        if not moving_vehicles:
            return events

        # Estimate dominant travel headings
        headings = []
        for t in moving_vehicles:
            for p in t.points:
                if p.heading_deg is not None:
                    headings.append(p.heading_deg)

        if not headings:
            return events

        counter = 0
        for t in moving_vehicles:
            t_headings = [p.heading_deg for p in t.points if p.heading_deg is not None]
            if not t_headings:
                continue
            mean_h = float(np.mean(t_headings))

            # Compare against opposing majority
            opposing_count = sum(
                1 for h in headings if calculate_angle_between_headings(mean_h, h) >= self.angle_threshold
            )
            # If over 70% of moving vehicles have opposing heading in similar spatial region
            if opposing_count > len(headings) * 0.4:
                counter += 1
                mid_pt = t.points[len(t.points) // 2]
                pos_w = (mid_pt.world_x_m, mid_pt.world_y_m) if mid_pt.world_x_m else None
                pos_px = (mid_pt.pixel_x, mid_pt.pixel_y)

                events.append(
                    TrafficEvent(
                        event_id=f"WRONGWAY_{counter:05d}",
                        event_type="WRONG_WAY",
                        start_timestamp_s=t.start_timestamp_s,
                        end_timestamp_s=t.end_timestamp_s,
                        start_frame=t.start_frame,
                        end_frame=t.end_frame,
                        involved_track_ids=[t.track_id],
                        primary_class=t.class_name,
                        severity=SeverityLevel.CRITICAL,
                        confidence=0.91,
                        location_world=pos_w,
                        location_pixel=pos_px,
                        description=f"Track #{t.track_id} ({t.class_name.value}) moved counter to dominant corridor orientation (Heading: {mean_h:.1f}°).",
                        metrics={"heading_deg": mean_h, "speed_kmh": t.average_speed_kmh},
                    )
                )

        return events
