"""
Pedestrian Crosswalk / Mid-Block Conflict Event Detector.
"""

from __future__ import annotations

from typing import List
import numpy as np
from traffic_intelligence.events.base import BaseEventDetector
from traffic_intelligence.schema import RoadUserClass, SeverityLevel, TrafficEvent, Trajectory


class PedestrianConflictEventDetector(BaseEventDetector):
    """Identifies pedestrian road-crossing events and vulnerable road user (VRU) encounters."""

    def detect(self, trajectories: List[Trajectory]) -> List[TrafficEvent]:
        events: List[TrafficEvent] = []
        counter = 0

        peds = [t for t in trajectories if t.class_name == RoadUserClass.PEDESTRIAN]
        vehicles = [t for t in trajectories if t.class_name != RoadUserClass.PEDESTRIAN]

        for ped in peds:
            ped_pts = ped.get_world_coordinates_array()
            if len(ped_pts) == 0:
                continue

            for veh in vehicles:
                # Check temporal overlap
                if not (ped.start_timestamp_s <= veh.end_timestamp_s and veh.start_timestamp_s <= ped.end_timestamp_s):
                    continue

                veh_pts = veh.get_world_coordinates_array()
                # Find min distance
                diffs = ped_pts[:, np.newaxis, :] - veh_pts[np.newaxis, :, :]
                dists = np.hypot(diffs[:, :, 0], diffs[:, :, 1])
                min_dist = float(np.min(dists))

                if min_dist < 6.0:  # VRU close proximity
                    counter += 1
                    mid_pt = ped.points[len(ped.points) // 2]
                    events.append(
                        TrafficEvent(
                            event_id=f"VRU_{counter:05d}",
                            event_type="PEDESTRIAN_VEHICLE_CONFLICT",
                            start_timestamp_s=ped.start_timestamp_s,
                            end_timestamp_s=ped.end_timestamp_s,
                            start_frame=ped.start_frame,
                            end_frame=ped.end_frame,
                            involved_track_ids=[ped.track_id, veh.track_id],
                            primary_class=RoadUserClass.PEDESTRIAN,
                            severity=SeverityLevel.CRITICAL if min_dist < 2.5 else SeverityLevel.HIGH,
                            confidence=0.94,
                            location_world=(mid_pt.world_x_m, mid_pt.world_y_m) if mid_pt.world_x_m else None,
                            location_pixel=(mid_pt.pixel_x, mid_pt.pixel_y),
                            description=f"Pedestrian #{ped.track_id} interacted closely with Vehicle #{veh.track_id} (Separation: {min_dist:.2f}m).",
                            metrics={"min_separation_m": min_dist, "vehicle_class": veh.class_name.value},
                        )
                    )

        return events
