"""
Core data schemas, enums, and domain representations for the Traffic Intelligence platform.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class RoadUserClass(str, Enum):
    CAR = "CAR"
    LGV = "LGV"
    HGV = "HGV"
    BUS = "BUS"
    TRUCK = "TRUCK"
    MOTORCYCLE = "MOTORCYCLE"
    PEDESTRIAN = "PEDESTRIAN"
    UNKNOWN = "UNKNOWN"


class TrackStateEnum(str, Enum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    OCCLUDED = "OCCLUDED"
    LOST = "LOST"
    REIDENTIFIED = "REIDENTIFIED"
    TERMINATED = "TERMINATED"


class MovementType(str, Enum):
    STRAIGHT = "STRAIGHT"
    LEFT_TURN = "LEFT_TURN"
    RIGHT_TURN = "RIGHT_TURN"
    U_TURN = "U_TURN"
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    STOPPED = "STOPPED"
    UNKNOWN = "UNKNOWN"


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CongestionLevel(str, Enum):
    FREE_FLOW = "FREE_FLOW"
    MILD = "MILD"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"
    STOPPED = "STOPPED"


class TelemetryRecord(BaseModel):
    """Drone flight telemetry sample at a specific timestamp."""
    timestamp_s: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude_m: Optional[float] = None
    heading_deg: Optional[float] = None
    gimbal_pitch_deg: Optional[float] = None
    gimbal_roll_deg: Optional[float] = None
    gimbal_yaw_deg: Optional[float] = None
    camera_fov_deg: Optional[float] = None
    drone_speed_mps: Optional[float] = None
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)


class VideoMetadata(BaseModel):
    """Metadata describing the ingested video or image sequence."""
    source_path: str
    fps: float
    width: int
    height: int
    total_frames: int
    duration_s: float
    codec: Optional[str] = None
    has_telemetry: bool = False
    telemetry_samples_count: int = 0
    estimated_gsd_m: Optional[float] = None  # Ground Sampling Distance (meters / pixel)


class Detection(BaseModel):
    """Object detection instance on a single frame."""
    frame_id: int
    timestamp_s: float
    bbox_xyxy: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax) in pixels
    confidence: float
    class_id: int
    raw_class_name: str
    standard_class: RoadUserClass = RoadUserClass.UNKNOWN
    feature_embedding: Optional[List[float]] = None

    @property
    def center_xy(self) -> Tuple[float, float]:
        xmin, ymin, xmax, ymax = self.bbox_xyxy
        return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)

    @property
    def width(self) -> float:
        return self.bbox_xyxy[2] - self.bbox_xyxy[0]

    @property
    def height(self) -> float:
        return self.bbox_xyxy[3] - self.bbox_xyxy[1]

    @property
    def area(self) -> float:
        return self.width * self.height


class Track(BaseModel):
    """Single tracked object observation in a single frame."""
    track_id: int
    frame_id: int
    timestamp_s: float
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    class_name: RoadUserClass
    state: TrackStateEnum
    world_x_m: Optional[float] = None
    world_y_m: Optional[float] = None
    velocity_x_mps: Optional[float] = None
    velocity_y_mps: Optional[float] = None
    speed_mps: Optional[float] = None
    heading_deg: Optional[float] = None
    feature_embedding: Optional[List[float]] = None

    @property
    def center_xy(self) -> Tuple[float, float]:
        xmin, ymin, xmax, ymax = self.bbox_xyxy
        return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)


class TrajectoryPoint(BaseModel):
    """Single state point along a smoothed, physical trajectory."""
    frame_id: int
    timestamp_s: float
    pixel_x: float
    pixel_y: float
    bbox_xyxy: Tuple[float, float, float, float]
    world_x_m: Optional[float] = None
    world_y_m: Optional[float] = None
    velocity_x_mps: Optional[float] = None
    velocity_y_mps: Optional[float] = None
    speed_mps: Optional[float] = None
    speed_kmh: Optional[float] = None
    acceleration_x_mps2: Optional[float] = None
    acceleration_y_mps2: Optional[float] = None
    acceleration_magnitude_mps2: Optional[float] = None
    jerk_mps3: Optional[float] = None
    heading_deg: Optional[float] = None
    angular_velocity_degps: Optional[float] = None
    confidence: float = 1.0


class Trajectory(BaseModel):
    """Complete temporal trajectory for a road user across its observed lifetime."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    track_id: int
    class_name: RoadUserClass
    start_frame: int
    end_frame: int
    start_timestamp_s: float
    end_timestamp_s: float
    points: List[TrajectoryPoint]
    movement_type: MovementType = MovementType.UNKNOWN
    origin_zone: Optional[str] = None
    destination_zone: Optional[str] = None

    # Quality & diagnostic metrics
    is_valid: bool = True
    quality_score: float = 1.0  # 0.0 to 1.0
    missing_ratio: float = 0.0
    total_distance_m: float = 0.0
    average_speed_kmh: float = 0.0
    max_speed_kmh: float = 0.0
    max_acceleration_mps2: float = 0.0
    max_deceleration_mps2: float = 0.0
    dwell_time_s: float = 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_timestamp_s - self.start_timestamp_s)

    @property
    def frame_count(self) -> int:
        return len(self.points)

    def get_world_coordinates_array(self) -> np.ndarray:
        coords = []
        for p in self.points:
            if p.world_x_m is not None and p.world_y_m is not None:
                coords.append([p.world_x_m, p.world_y_m])
            else:
                coords.append([p.pixel_x, p.pixel_y])
        return np.array(coords, dtype=np.float64)

    def get_pixel_coordinates_array(self) -> np.ndarray:
        return np.array([[p.pixel_x, p.pixel_y] for p in self.points], dtype=np.float64)


class ConflictEvent(BaseModel):
    """Surrogate safety conflict event between two or more interacting road users."""
    event_id: str
    timestamp_s: float
    frame_id: int
    primary_track_id: int
    secondary_track_id: int
    primary_class: RoadUserClass
    secondary_class: RoadUserClass
    conflict_type: str  # "REAR_END", "CROSSING", "MERGING", "TURNING", "PEDESTRIAN_VEHICLE", "MOTORCYCLE_VEHICLE"
    severity: SeverityLevel
    ttc_s: Optional[float] = None
    pet_s: Optional[float] = None
    drac_mps2: Optional[float] = None
    separation_distance_m: float
    relative_speed_mps: float
    conflict_angle_deg: Optional[float] = None
    location_world: Optional[Tuple[float, float]] = None
    location_pixel: Tuple[float, float]
    metrics: Dict[str, Any] = Field(default_factory=dict)


class TrafficEvent(BaseModel):
    """Behavioral or structural traffic event detected from trajectories."""
    event_id: str
    event_type: str  # "SUDDEN_BRAKING", "SUDDEN_ACCELERATION", "WRONG_WAY", "STOPPED_VEHICLE", "CUT_IN", "UNSAFE_MERGE", "ANOMALY"
    start_timestamp_s: float
    end_timestamp_s: float
    start_frame: int
    end_frame: int
    involved_track_ids: List[int]
    primary_class: RoadUserClass
    severity: SeverityLevel
    confidence: float
    location_world: Optional[Tuple[float, float]] = None
    location_pixel: Tuple[float, float]
    description: str
    metrics: Dict[str, Any] = Field(default_factory=dict)


class QueueState(BaseModel):
    """Temporal snapshot of a spatial queue formed along a movement/lane."""
    queue_id: str
    timestamp_s: float
    frame_id: int
    lane_id: Optional[str] = None
    vehicle_count: int
    queue_length_m: float
    max_queue_length_m: float
    duration_s: float
    growth_rate_mps: float
    dissipation_rate_mps: float
    involved_track_ids: List[int]
    head_location_world: Optional[Tuple[float, float]] = None
    tail_location_world: Optional[Tuple[float, float]] = None
