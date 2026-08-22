"""
Spatial-Temporal Interaction Graph: models pairwise road user influences and relative dynamics.
Uses Scipy KDTree for efficient O(N log N) spatial neighborhood queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial import KDTree

from traffic_intelligence.geometry.coordinates import calculate_angle_between_headings
from traffic_intelligence.schema import RoadUserClass, TrajectoryPoint
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.interaction_graph")


@dataclass
class AgentInteractionEdge:
    """Directed edge representing influence/interaction from agent_a to agent_b."""
    timestamp_s: float
    frame_id: int
    track_id_a: int
    track_id_b: int
    class_a: RoadUserClass
    class_b: RoadUserClass
    distance_m: float
    relative_speed_mps: float
    relative_heading_deg: float
    closing_speed_mps: float
    pos_a: Tuple[float, float]
    pos_b: Tuple[float, float]
    vel_a: Tuple[float, float]
    vel_b: Tuple[float, float]


class InteractionGraph:
    """Builds spatial proximity interaction graphs for all road users at each timestamp."""

    def __init__(self, influence_radius_m: float = 25.0):
        self.influence_radius_m = influence_radius_m

    def build_graph_at_frame(
        self,
        frame_id: int,
        timestamp_s: float,
        active_agents: List[Tuple[int, RoadUserClass, TrajectoryPoint]],
    ) -> List[AgentInteractionEdge]:
        """
        Constructs interaction edges for all pairs within influence_radius_m at a given frame.
        active_agents: List of (track_id, class_name, TrajectoryPoint).
        """
        N = len(active_agents)
        if N < 2:
            return []

        # Extract positions
        coords = []
        for _, _, pt in active_agents:
            x = pt.world_x_m if pt.world_x_m is not None else pt.pixel_x
            y = pt.world_y_m if pt.world_y_m is not None else pt.pixel_y
            coords.append([x, y])

        coords_arr = np.array(coords, dtype=np.float64)
        tree = KDTree(coords_arr)
        # Query all pairs within influence radius
        pairs = tree.query_pairs(r=self.influence_radius_m)

        edges: List[AgentInteractionEdge] = []
        for i, j in pairs:
            tid_a, cls_a, pt_a = active_agents[i]
            tid_b, cls_b, pt_b = active_agents[j]

            pos_a = (coords_arr[i, 0], coords_arr[i, 1])
            pos_b = (coords_arr[j, 0], coords_arr[j, 1])

            dist = float(np.hypot(pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]))

            vx_a = pt_a.velocity_x_mps or 0.0
            vy_a = pt_a.velocity_y_mps or 0.0
            vx_b = pt_b.velocity_x_mps or 0.0
            vy_b = pt_b.velocity_y_mps or 0.0

            rel_vx = vx_b - vx_a
            rel_vy = vy_b - vy_a
            rel_speed = float(np.hypot(rel_vx, rel_vy))

            # Closing speed: component of relative velocity along vector from A to B
            # r_vec = B - A
            rx, ry = pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]
            if dist > 1e-4:
                # closing_speed = - d(dist)/dt
                closing_speed = float(-(rx * rel_vx + ry * rel_vy) / dist)
            else:
                closing_speed = 0.0

            head_a = pt_a.heading_deg or 0.0
            head_b = pt_b.heading_deg or 0.0
            rel_heading = calculate_angle_between_headings(head_a, head_b)

            edge = AgentInteractionEdge(
                timestamp_s=timestamp_s,
                frame_id=frame_id,
                track_id_a=tid_a,
                track_id_b=tid_b,
                class_a=cls_a,
                class_b=cls_b,
                distance_m=dist,
                relative_speed_mps=rel_speed,
                relative_heading_deg=rel_heading,
                closing_speed_mps=closing_speed,
                pos_a=pos_a,
                pos_b=pos_b,
                vel_a=(vx_a, vy_a),
                vel_b=(vx_b, vy_b),
            )
            edges.append(edge)

        return edges
