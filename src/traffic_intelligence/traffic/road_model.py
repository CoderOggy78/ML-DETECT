"""
Road and Zone models using Shapely geometry polygons.
Supports user-defined entry/exit zones and automatic spatial bounding.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from shapely.geometry import Point, Polygon

from traffic_intelligence.schema import Trajectory
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.road_model")


class ZoneModel:
    """Represents a polygonal spatial zone (e.g. entry approach, exit arm, conflict zone, stop line)."""

    def __init__(self, name: str, polygon_coords: List[Tuple[float, float]], zone_type: str = "general"):
        self.name = name
        self.zone_type = zone_type  # "entry", "exit", "conflict", "general"
        self.polygon = Polygon(polygon_coords)

    def contains_point(self, x: float, y: float) -> bool:
        return self.polygon.contains(Point(x, y))

    def intersects_trajectory(self, trajectory: Trajectory) -> bool:
        pts = trajectory.get_world_coordinates_array()
        for p in pts:
            if self.contains_point(p[0], p[1]):
                return True
        return False


class RoadModel:
    """Road network representation containing collection of zones and boundary polygons."""

    def __init__(self, zones: Optional[List[ZoneModel]] = None):
        self.zones: List[ZoneModel] = zones or []

    def add_zone(self, zone: ZoneModel) -> None:
        self.zones.append(zone)

    def get_zone_by_name(self, name: str) -> Optional[ZoneModel]:
        for z in self.zones:
            if z.name.lower() == name.lower():
                return z
        return None

    def identify_entry_exit(self, trajectory: Trajectory) -> Tuple[Optional[str], Optional[str]]:
        """Identifies origin (entry) and destination (exit) zones for a given trajectory."""
        pts = trajectory.get_world_coordinates_array()
        if len(pts) == 0:
            return None, None

        start_pt = pts[0]
        end_pt = pts[-1]

        origin: Optional[str] = None
        destination: Optional[str] = None

        for z in self.zones:
            if z.contains_point(start_pt[0], start_pt[1]):
                origin = z.name
            if z.contains_point(end_pt[0], end_pt[1]):
                destination = z.name

        return origin, destination

    @classmethod
    def auto_discover_from_trajectories(
        cls, trajectories: List[Trajectory], margin_ratio: float = 0.15
    ) -> RoadModel:
        """
        Unsupervised discovery of entry/exit zones by clustering start and end coordinates of trajectories.
        Does not require manual annotations!
        """
        if not trajectories:
            return cls()

        start_pts = []
        end_pts = []
        for t in trajectories:
            wpts = t.get_world_coordinates_array()
            if len(wpts) > 0:
                start_pts.append(wpts[0])
                end_pts.append(wpts[-1])

        all_pts = np.vstack(start_pts + end_pts)
        min_x, min_y = np.min(all_pts, axis=0)
        max_x, max_y = np.max(all_pts, axis=0)
        dx, dy = max_x - min_x, max_y - min_y

        # Create 4 cardinal boundary zones (North, South, East, West approaches)
        zones = [
            ZoneModel(
                "West_Approach",
                [(min_x - 5, min_y), (min_x + dx * margin_ratio, min_y), (min_x + dx * margin_ratio, max_y), (min_x - 5, max_y)],
                zone_type="boundary",
            ),
            ZoneModel(
                "East_Approach",
                [(max_x - dx * margin_ratio, min_y), (max_x + 5, min_y), (max_x + 5, max_y), (max_x - dx * margin_ratio, max_y)],
                zone_type="boundary",
            ),
            ZoneModel(
                "North_Approach",
                [(min_x, max_y - dy * margin_ratio), (max_x, max_y - dy * margin_ratio), (max_x, max_y + 5), (min_x, max_y + 5)],
                zone_type="boundary",
            ),
            ZoneModel(
                "South_Approach",
                [(min_x, min_y - 5), (max_x, min_y - 5), (max_x, min_y + dy * margin_ratio), (min_x, min_y + dy * margin_ratio)],
                zone_type="boundary",
            ),
        ]
        logger.info(f"Auto-generated {len(zones)} road boundary zones from {len(trajectories)} trajectories.")
        return cls(zones=zones)
