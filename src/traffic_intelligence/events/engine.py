"""
Event Engine: orchestrates pluggable event detectors and produces standardized traffic event logs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd

from traffic_intelligence.events.acceleration import SuddenAccelerationDetector
from traffic_intelligence.events.base import BaseEventDetector
from traffic_intelligence.events.braking import SuddenBrakingDetector
from traffic_intelligence.events.cut_in import CutInDetector
from traffic_intelligence.events.pedestrian import PedestrianConflictEventDetector
from traffic_intelligence.events.stopped import StoppedVehicleDetector
from traffic_intelligence.events.wrong_way import WrongWayDetector
from traffic_intelligence.schema import TrafficEvent, Trajectory
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.events.engine")


class EventEngine:
    """Dispatches trajectory data across behavioral event detectors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        evt_cfg = config.get("events", {}) if config else {}

        self.detectors: List[BaseEventDetector] = [
            SuddenBrakingDetector(
                hard_braking_threshold_mps2=evt_cfg.get("hard_braking_threshold_mps2", -4.5)
            ),
            SuddenAccelerationDetector(
                hard_accel_threshold_mps2=evt_cfg.get("hard_accel_threshold_mps2", 3.5)
            ),
            StoppedVehicleDetector(
                min_dwell_time_s=evt_cfg.get("stopped_vehicle_duration_s", 5.0)
            ),
            WrongWayDetector(
                wrong_way_angle_threshold_deg=evt_cfg.get("wrong_way_angle_threshold_deg", 120.0)
            ),
            CutInDetector(),
            PedestrianConflictEventDetector(),
        ]

    def register_detector(self, detector: BaseEventDetector) -> None:
        self.detectors.append(detector)

    def detect_all(self, trajectories: List[Trajectory]) -> List[TrafficEvent]:
        """Runs all registered event detectors against the trajectory corpus."""
        all_events: List[TrafficEvent] = []
        for det in self.detectors:
            try:
                evts = det.detect(trajectories)
                all_events.extend(evts)
            except Exception as e:
                logger.warning(f"Event detector {det.__class__.__name__} failed: {e}")

        logger.info(f"EventEngine extracted {len(all_events)} domain traffic events.")
        return all_events

    @staticmethod
    def events_to_dataframe(events: List[TrafficEvent]) -> pd.DataFrame:
        """Converts TrafficEvent list to a pandas DataFrame."""
        rows = []
        for e in events:
            rows.append(
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type,
                    "start_frame": e.start_frame,
                    "end_frame": e.end_frame,
                    "start_timestamp_s": e.start_timestamp_s,
                    "end_timestamp_s": e.end_timestamp_s,
                    "involved_tracks": ",".join(map(str, e.involved_track_ids)),
                    "primary_class": e.primary_class.value,
                    "severity": e.severity.value,
                    "confidence": e.confidence,
                    "world_x_m": e.location_world[0] if e.location_world else None,
                    "world_y_m": e.location_world[1] if e.location_world else None,
                    "description": e.description,
                }
            )
        return pd.DataFrame(rows)
