"""
Surrogate Safety Conflict Detector: classifies interactions into safety conflicts with severity levels.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import pandas as pd

from traffic_intelligence.interactions.drac import calculate_deceleration_rate_to_avoid_collision
from traffic_intelligence.interactions.graph import AgentInteractionEdge, InteractionGraph
from traffic_intelligence.interactions.pet import PostEncroachmentTimeCalculator
from traffic_intelligence.interactions.ttc import calculate_time_to_collision
from traffic_intelligence.schema import ConflictEvent, RoadUserClass, SeverityLevel, Trajectory, TrajectoryPoint
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.conflicts")


class ConflictDetector:
    """Detects surrogate safety events / near-miss potential conflicts from multi-agent trajectories."""

    def __init__(
        self,
        ttc_threshold_critical_s: float = 1.5,
        ttc_threshold_warning_s: float = 2.5,
        pet_threshold_critical_s: float = 1.0,
        pet_threshold_warning_s: float = 2.0,
        drac_threshold_critical_mps2: float = 4.0,
        drac_threshold_warning_mps2: float = 2.5,
        influence_radius_m: float = 25.0,
    ):
        self.ttc_critical = ttc_threshold_critical_s
        self.ttc_warning = ttc_threshold_warning_s
        self.pet_critical = pet_threshold_critical_s
        self.pet_warning = pet_threshold_warning_s
        self.drac_critical = drac_threshold_critical_mps2
        self.drac_warning = drac_threshold_warning_mps2

        self.graph_builder = InteractionGraph(influence_radius_m=influence_radius_m)
        self.pet_calculator = PostEncroachmentTimeCalculator()

    def classify_conflict_type(
        self, class_a: RoadUserClass, class_b: RoadUserClass, rel_heading_deg: float
    ) -> str:
        """Categorizes conflict topology based on road-user classes and collision angle."""
        if class_a == RoadUserClass.PEDESTRIAN or class_b == RoadUserClass.PEDESTRIAN:
            return "PEDESTRIAN_VEHICLE"
        if class_a == RoadUserClass.MOTORCYCLE or class_b == RoadUserClass.MOTORCYCLE:
            return "MOTORCYCLE_VEHICLE"

        # Vehicle-vehicle conflicts by angle:
        # < 35 deg: Rear-end / Front-to-back
        # 35 - 85 deg: Merging / Cut-in
        # > 85 deg: Crossing / Intersection
        if rel_heading_deg < 35.0:
            return "REAR_END"
        elif 35.0 <= rel_heading_deg < 85.0:
            return "MERGING"
        elif 85.0 <= rel_heading_deg < 145.0:
            return "CROSSING"
        else:
            return "HEAD_ON"

    def evaluate_severity(
        self, ttc: Optional[float], pet: Optional[float], drac: Optional[float]
    ) -> SeverityLevel:
        """Determines conflict severity level from TTC, PET, and DRAC values."""
        is_critical = (
            (ttc is not None and ttc <= self.ttc_critical)
            or (pet is not None and pet <= self.pet_critical)
            or (drac is not None and drac >= self.drac_critical)
        )
        if is_critical:
            return SeverityLevel.CRITICAL

        is_high = (
            (ttc is not None and ttc <= (self.ttc_critical + self.ttc_warning) / 2.0)
            or (pet is not None and pet <= (self.pet_critical + self.pet_warning) / 2.0)
            or (drac is not None and drac >= (self.drac_critical + self.drac_warning) / 2.0)
        )
        if is_high:
            return SeverityLevel.HIGH

        is_medium = (
            (ttc is not None and ttc <= self.ttc_warning)
            or (pet is not None and pet <= self.pet_warning)
            or (drac is not None and drac >= self.drac_warning)
        )
        if is_medium:
            return SeverityLevel.MEDIUM

        return SeverityLevel.LOW

    def detect_conflicts(self, trajectories: List[Trajectory]) -> List[ConflictEvent]:
        """Runs conflict detection across entire trajectory corpus."""
        # Index trajectory points by frame_id
        frame_map: Dict[int, List[Tuple[int, RoadUserClass, TrajectoryPoint]]] = {}
        for t in trajectories:
            for p in t.points:
                if p.frame_id not in frame_map:
                    frame_map[p.frame_id] = []
                frame_map[p.frame_id].append((t.track_id, t.class_name, p))

        events: List[ConflictEvent] = []
        event_counter = 0

        for fid in sorted(frame_map.keys()):
            agents = frame_map[fid]
            ts = agents[0][2].timestamp_s
            edges: List[AgentInteractionEdge] = self.graph_builder.build_graph_at_frame(fid, ts, agents)

            for edge in edges:
                # Compute TTC
                ttc = calculate_time_to_collision(
                    edge.pos_a, edge.vel_a, edge.pos_b, edge.vel_b, collision_radius_m=3.0
                )

                # Compute DRAC
                drac = calculate_deceleration_rate_to_avoid_collision(
                    edge.distance_m, edge.closing_speed_mps
                )

                # Filter for meaningful interactions (either small TTC or significant DRAC or close proximity)
                is_conflict_candidate = (
                    (ttc is not None and ttc <= self.ttc_warning)
                    or (drac is not None and drac >= self.drac_warning)
                    or (edge.distance_m < 3.5 and edge.relative_speed_mps > 2.0)
                )

                if is_conflict_candidate:
                    c_type = self.classify_conflict_type(edge.class_a, edge.class_b, edge.relative_heading_deg)
                    severity = self.evaluate_severity(ttc, None, drac)

                    event_counter += 1
                    mid_pixel = (
                        (edge.pos_a[0] + edge.pos_b[0]) / 2.0,
                        (edge.pos_a[1] + edge.pos_b[1]) / 2.0,
                    )
                    event = ConflictEvent(
                        event_id=f"CONF_{event_counter:05d}",
                        timestamp_s=edge.timestamp_s,
                        frame_id=edge.frame_id,
                        primary_track_id=edge.track_id_a,
                        secondary_track_id=edge.track_id_b,
                        primary_class=edge.class_a,
                        secondary_class=edge.class_b,
                        conflict_type=c_type,
                        severity=severity,
                        ttc_s=ttc,
                        drac_mps2=drac,
                        separation_distance_m=edge.distance_m,
                        relative_speed_mps=edge.relative_speed_mps,
                        conflict_angle_deg=edge.relative_heading_deg,
                        location_world=mid_pixel,
                        location_pixel=mid_pixel,
                        metrics={
                            "closing_speed_mps": edge.closing_speed_mps,
                            "distance_m": edge.distance_m,
                        },
                    )
                    events.append(event)

        logger.info(f"Identified {len(events)} surrogate safety conflict events.")
        return events

    @staticmethod
    def conflicts_to_dataframe(conflicts: List[ConflictEvent]) -> pd.DataFrame:
        """Converts ConflictEvent list to a flat pandas DataFrame for CSV and analysis export."""
        rows = []
        for c in conflicts:
            rows.append(
                {
                    "event_id": c.event_id,
                    "frame_id": c.frame_id,
                    "timestamp_s": c.timestamp_s,
                    "primary_track_id": c.primary_track_id,
                    "secondary_track_id": c.secondary_track_id,
                    "primary_class": c.primary_class.value,
                    "secondary_class": c.secondary_class.value,
                    "conflict_type": c.conflict_type,
                    "severity": c.severity.value,
                    "ttc_s": c.ttc_s,
                    "drac_mps2": c.drac_mps2,
                    "separation_distance_m": c.separation_distance_m,
                    "relative_speed_mps": c.relative_speed_mps,
                    "conflict_angle_deg": c.conflict_angle_deg,
                    "world_x_m": c.location_world[0] if c.location_world else None,
                    "world_y_m": c.location_world[1] if c.location_world else None,
                }
            )
        return pd.DataFrame(rows)
