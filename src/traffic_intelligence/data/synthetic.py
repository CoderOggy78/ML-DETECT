"""
Synthetic Drone Traffic Generator: produces realistic aerial video, telemetry, and trajectories.
Simulates intersection crossing, rear-end braking, cut-ins, queues, wrong-way, and pedestrian conflicts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd

from traffic_intelligence.schema import (
    Detection,
    MovementType,
    RoadUserClass,
    Track,
    TrackStateEnum,
    Trajectory,
    TrajectoryPoint,
)
from traffic_intelligence.trajectories.motion import MotionEstimator
from traffic_intelligence.trajectories.quality import TrajectoryQualityEvaluator
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.synthetic")


class SyntheticTrafficGenerator:
    """Generates synthetic aerial traffic scenes with ground-truth physics for validation."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        duration_s: float = 10.0,
        meters_per_pixel: float = 0.05,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.duration_s = duration_s
        self.total_frames = int(fps * duration_s)
        self.meters_per_pixel = meters_per_pixel

    def generate_scenario(
        self, output_video_path: Optional[Path] = None, output_telemetry_path: Optional[Path] = None
    ) -> Tuple[List[Trajectory], List[Dict[str, Any]]]:
        """Creates realistic multi-agent trajectories and optionally renders an MP4 video."""
        trajectories: List[Trajectory] = []
        raw_detections: List[Dict[str, Any]] = []

        # Scenario Agents:
        # 1. Car A: Eastbound straight (normal speed)
        # 2. Car B: Eastbound behind Car A, experiences sudden braking
        # 3. Truck C: Northbound straight crossing intersection
        # 4. Car D: Westbound making a Left Turn
        # 5. Pedestrian E: Crossing south-north on crosswalk
        # 6. Motorcycle F: Cut-in / aggressive lane change
        # 7. Car G: Queue vehicle stopped at intersection
        # 8. Car H: Wrong-way vehicle moving opposite to eastbound lane

        agents = [
            {
                "id": 1,
                "class": RoadUserClass.CAR,
                "type": MovementType.STRAIGHT,
                "color": (255, 100, 100),
                "box_size": (40, 20),
                "start": (100, 340),
                "end": (1180, 340),
                "speed_base": 14.0,  # ~50 km/h
                "behavior": "normal",
            },
            {
                "id": 2,
                "class": RoadUserClass.CAR,
                "type": MovementType.STRAIGHT,
                "color": (100, 100, 255),
                "box_size": (40, 20),
                "start": (40, 340),
                "end": (900, 340),
                "speed_base": 15.0,
                "behavior": "sudden_brake",
            },
            {
                "id": 3,
                "class": RoadUserClass.TRUCK,
                "type": MovementType.STRAIGHT,
                "color": (100, 200, 100),
                "box_size": (70, 25),
                "start": (620, 50),
                "end": (620, 670),
                "speed_base": 10.0,
                "behavior": "normal",
            },
            {
                "id": 4,
                "class": RoadUserClass.CAR,
                "type": MovementType.LEFT_TURN,
                "color": (220, 220, 50),
                "box_size": (40, 20),
                "start": (1180, 380),
                "end": (620, 680),
                "speed_base": 9.0,
                "behavior": "turn_left",
            },
            {
                "id": 5,
                "class": RoadUserClass.PEDESTRIAN,
                "type": MovementType.STRAIGHT,
                "color": (255, 50, 255),
                "box_size": (12, 12),
                "start": (560, 460),
                "end": (560, 260),
                "speed_base": 1.4,  # ~5 km/h walking
                "behavior": "crossing",
            },
            {
                "id": 6,
                "class": RoadUserClass.MOTORCYCLE,
                "type": MovementType.STRAIGHT,
                "color": (50, 230, 230),
                "box_size": (22, 12),
                "start": (80, 300),
                "end": (1150, 345),
                "speed_base": 18.0,
                "behavior": "cut_in",
            },
            {
                "id": 7,
                "class": RoadUserClass.LGV,
                "type": MovementType.STOPPED,
                "color": (150, 150, 150),
                "box_size": (45, 22),
                "start": (500, 340),
                "end": (505, 340),
                "speed_base": 0.0,
                "behavior": "stopped",
            },
            {
                "id": 8,
                "class": RoadUserClass.CAR,
                "type": MovementType.STRAIGHT,
                "color": (255, 0, 0),
                "box_size": (40, 20),
                "start": (1100, 340),
                "end": (200, 340),
                "speed_base": 12.0,
                "behavior": "wrong_way",
            },
        ]

        # Simulate frame-by-frame
        frames_data: Dict[int, List[Dict[str, Any]]] = {f: [] for f in range(self.total_frames)}

        for agent in agents:
            pts: List[TrajectoryPoint] = []
            aid = agent["id"]
            acls = agent["class"]
            w_box, h_box = agent["box_size"]
            bhv = agent["behavior"]

            x_curr, y_curr = float(agent["start"][0]), float(agent["start"][1])
            x_end, y_end = float(agent["end"][0]), float(agent["end"][1])

            for f in range(self.total_frames):
                t_s = f / self.fps

                # Movement dynamics based on behavior
                if bhv == "normal":
                    if aid == 1:
                        dx = 4.0 * (agent["speed_base"] / 10.0)
                        x_curr += dx
                        y_curr += 0.05 * math.sin(f * 0.1)
                    elif aid == 3:
                        dy = 3.0 * (agent["speed_base"] / 10.0)
                        y_curr += dy

                elif bhv == "sudden_brake":
                    if f < 60:
                        x_curr += 10.0
                    elif 60 <= f < 80:  # Sudden hard braking
                        x_curr += 1.0
                    else:  # Recovery
                        x_curr += 5.0

                elif bhv == "turn_left":
                    if x_curr > 640:
                        x_curr -= 6.0
                    else:
                        y_curr += 6.0
                        x_curr -= 0.5 if x_curr > 620 else 0.0

                elif bhv == "crossing":
                    y_curr -= 1.5  # Walking across street

                elif bhv == "cut_in":
                    x_curr += 8.0
                    if 40 <= f <= 100:
                        y_curr += 1.5  # Swerve into lane

                elif bhv == "stopped":
                    pass  # Perfectly stationary

                elif bhv == "wrong_way":
                    x_curr -= 8.0  # Moving left in eastbound lane

                # Boundary check
                if not (0 <= x_curr <= self.width and 0 <= y_curr <= self.height):
                    continue

                # Add camera jitter simulation (drone wind effect) unless stationary
                if bhv == "stopped":
                    jitter_x = 0.0
                    jitter_y = 0.0
                else:
                    jitter_x = 1.0 * math.sin(f * 0.2)
                    jitter_y = 1.0 * math.cos(f * 0.15)

                obs_x = x_curr + jitter_x
                obs_y = y_curr + jitter_y

                xmin = obs_x - w_box / 2.0
                ymin = obs_y - h_box / 2.0
                xmax = obs_x + w_box / 2.0
                ymax = obs_y + h_box / 2.0

                world_x = obs_x * self.meters_per_pixel
                world_y = obs_y * self.meters_per_pixel

                pt = TrajectoryPoint(
                    frame_id=f,
                    timestamp_s=t_s,
                    pixel_x=float(obs_x),
                    pixel_y=float(obs_y),
                    bbox_xyxy=(float(xmin), float(ymin), float(xmax), float(ymax)),
                    world_x_m=float(world_x),
                    world_y_m=float(world_y),
                    confidence=0.95,
                )
                pts.append(pt)

                det_dict = {
                    "frame_id": f,
                    "timestamp_s": t_s,
                    "track_id": aid,
                    "bbox": [xmin, ymin, xmax, ymax],
                    "confidence": 0.95,
                    "class_name": acls.value,
                    "color": agent["color"],
                }
                frames_data[f].append(det_dict)
                raw_detections.append(det_dict)

            if len(pts) > 5:
                enriched_pts = MotionEstimator.estimate_kinematics(pts)
                traj = Trajectory(
                    track_id=aid,
                    class_name=acls,
                    start_frame=enriched_pts[0].frame_id,
                    end_frame=enriched_pts[-1].frame_id,
                    start_timestamp_s=enriched_pts[0].timestamp_s,
                    end_timestamp_s=enriched_pts[-1].timestamp_s,
                    points=enriched_pts,
                    movement_type=agent["type"],
                    quality_score=0.98,
                )
                traj = TrajectoryQualityEvaluator.evaluate(traj, min_length_frames=5)
                trajectories.append(traj)

        # Render video if requested
        if output_video_path:
            self._render_video(output_video_path, frames_data)

        # Render telemetry if requested
        if output_telemetry_path:
            self._render_telemetry(output_telemetry_path)

        logger.info(
            f"Generated synthetic scenario: {len(trajectories)} trajectories across {self.total_frames} frames"
        )
        return trajectories, raw_detections

    def _render_video(self, output_path: Path, frames_data: Dict[int, List[Dict[str, Any]]]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))

        for f in range(self.total_frames):
            frame = np.full((self.height, self.width, 3), 50, dtype=np.uint8)  # Dark asphalt background

            # Draw Road Network (4-way intersection)
            # East-West road
            cv2.rectangle(frame, (0, 280), (self.width, 440), (70, 70, 70), -1)
            # North-South road
            cv2.rectangle(frame, (540, 0), (740, self.height), (70, 70, 70), -1)

            # Lane markings (dashed yellow & white lines)
            for x in range(0, self.width, 40):
                if not (540 <= x <= 740):
                    cv2.line(frame, (x, 360), (x + 20, 360), (0, 220, 220), 2)
            for y in range(0, self.height, 40):
                if not (280 <= y <= 440):
                    cv2.line(frame, (640, y), (640, y + 20), (0, 220, 220), 2)

            # Crosswalks
            for offset in range(550, 730, 20):
                cv2.rectangle(frame, (offset, 250), (offset + 10, 275), (200, 200, 200), -1)
                cv2.rectangle(frame, (offset, 445), (offset + 10, 470), (200, 200, 200), -1)

            # Draw vehicles / pedestrians
            for det in frames_data.get(f, []):
                xmin, ymin, xmax, ymax = map(int, det["bbox"])
                color = det["color"]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, -1)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
                cv2.putText(
                    frame,
                    f"ID:{det['track_id']} {det['class_name']}",
                    (xmin, max(15, ymin - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

            # Add frame timestamp header
            cv2.putText(
                frame,
                f"Synthetic Aerial Feed | Frame {f:04d} | Time: {f/self.fps:.2f}s",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            writer.write(frame)

        writer.release()
        logger.info(f"Rendered synthetic MP4 video: {output_path}")

    def _render_telemetry(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_rows = []
        for f in range(self.total_frames):
            t_s = f / self.fps
            alt = 60.0 + 0.5 * math.sin(f * 0.05)
            yaw = 45.0 + 0.2 * math.cos(f * 0.08)
            pitch = -88.5 + 0.1 * math.sin(f * 0.1)
            telemetry_rows.append(
                {
                    "timestamp_s": t_s,
                    "latitude": 37.7749 + (f * 0.000001),
                    "longitude": -122.4194 + (f * 0.000001),
                    "altitude_m": alt,
                    "heading_deg": yaw,
                    "gimbal_pitch_deg": pitch,
                    "gimbal_roll_deg": 0.1 * math.sin(f * 0.04),
                    "drone_speed_mps": 0.3 + 0.05 * math.cos(f * 0.02),
                }
            )
        pd.DataFrame(telemetry_rows).to_csv(output_path, index=False)
        logger.info(f"Rendered synthetic flight telemetry: {output_path}")
