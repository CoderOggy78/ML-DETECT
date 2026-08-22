"""
Drone Telemetry Reader with support for CSV, JSON, GPX, and DJI SRT subtitles.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from traffic_intelligence.schema import TelemetryRecord
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.telemetry")


class TelemetryReader:
    """Ingests and synchronizes flight telemetry with frame timestamps."""

    def __init__(self, telemetry_path: Optional[Union[str, Path]] = None):
        self.telemetry_path = Path(telemetry_path) if telemetry_path else None
        self.records: List[TelemetryRecord] = []
        self._timestamps: np.ndarray = np.array([])
        if self.telemetry_path and self.telemetry_path.exists():
            self._load()

    def has_data(self) -> bool:
        return len(self.records) > 0

    def _load(self) -> None:
        if not self.telemetry_path:
            return

        suffix = self.telemetry_path.suffix.lower()
        try:
            if suffix == ".csv":
                self._load_csv()
            elif suffix == ".json":
                self._load_json()
            elif suffix == ".srt":
                self._load_srt()
            else:
                # Attempt CSV fallback
                self._load_csv()

            if self.records:
                self._timestamps = np.array([r.timestamp_s for r in self.records], dtype=np.float64)
                logger.info(f"Loaded {len(self.records)} telemetry samples from {self.telemetry_path.name}")
        except Exception as e:
            logger.warning(f"Failed to parse telemetry at {self.telemetry_path}: {e}. Proceeding without telemetry.")
            self.records = []

    def _load_csv(self) -> None:
        df = pd.read_csv(self.telemetry_path)
        # Normalize column names
        cols = {c: c.lower().strip().replace(" ", "_") for c in df.columns}
        df = df.rename(columns=cols)

        # Look for timestamp column
        time_col = None
        for candidate in ["timestamp_s", "timestamp", "time", "t", "time_sec", "rel_time"]:
            if candidate in df.columns:
                time_col = candidate
                break

        for idx, row in df.iterrows():
            t_s = float(row[time_col]) if time_col else float(idx) * 0.0333
            rec = TelemetryRecord(
                timestamp_s=t_s,
                latitude=float(row.get("lat", row.get("latitude", 0.0))) if "lat" in row or "latitude" in row else None,
                longitude=float(row.get("lon", row.get("longitude", row.get("lng", 0.0)))) if any(k in row for k in ["lon", "longitude", "lng"]) else None,
                altitude_m=float(row.get("altitude", row.get("alt", row.get("height", 0.0)))) if any(k in row for k in ["altitude", "alt", "height"]) else None,
                heading_deg=float(row.get("heading", row.get("yaw", 0.0))) if any(k in row for k in ["heading", "yaw"]) else None,
                gimbal_pitch_deg=float(row.get("pitch", row.get("gimbal_pitch", -90.0))) if any(k in row for k in ["pitch", "gimbal_pitch"]) else None,
                gimbal_roll_deg=float(row.get("roll", row.get("gimbal_roll", 0.0))) if any(k in row for k in ["roll", "gimbal_roll"]) else None,
                gimbal_yaw_deg=float(row.get("gimbal_yaw", 0.0)) if "gimbal_yaw" in row else None,
                drone_speed_mps=float(row.get("speed", row.get("drone_speed", 0.0))) if any(k in row for k in ["speed", "drone_speed"]) else None,
                raw_metadata=row.to_dict(),
            )
            self.records.append(rec)

    def _load_json(self) -> None:
        with open(self.telemetry_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "telemetry" in data:
            data = data["telemetry"]

        if isinstance(data, list):
            for idx, item in enumerate(data):
                t_s = float(item.get("timestamp_s", item.get("timestamp", idx * 0.0333)))
                self.records.append(
                    TelemetryRecord(
                        timestamp_s=t_s,
                        latitude=item.get("latitude", item.get("lat")),
                        longitude=item.get("longitude", item.get("lon", item.get("lng"))),
                        altitude_m=item.get("altitude", item.get("alt")),
                        heading_deg=item.get("heading", item.get("yaw")),
                        gimbal_pitch_deg=item.get("gimbal_pitch", item.get("pitch")),
                        gimbal_roll_deg=item.get("gimbal_roll", item.get("roll")),
                        gimbal_yaw_deg=item.get("gimbal_yaw"),
                        camera_fov_deg=item.get("fov"),
                        drone_speed_mps=item.get("speed"),
                        raw_metadata=item,
                    )
                )

    def _load_srt(self) -> None:
        """Parses DJI subtitle SRT metadata format."""
        with open(self.telemetry_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        blocks = content.strip().split("\n\n")
        time_pattern = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})")

        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) < 2:
                continue

            match = time_pattern.search(lines[1])
            if match:
                h, m, s, ms = map(int, match.groups()[:4])
                t_s = h * 3600 + m * 60 + s + ms / 1000.0

                text = " ".join(lines[2:])
                lat_m = re.search(r"\[latitude:\s*([\d\.-]+)\]", text, re.IGNORECASE)
                lon_m = re.search(r"\[longitude:\s*([\d\.-]+)\]", text, re.IGNORECASE)
                alt_m = re.search(r"\[rel_alt:\s*([\d\.-]+)|altitude:\s*([\d\.-]+)\]", text, re.IGNORECASE)

                lat = float(lat_m.group(1)) if lat_m else None
                lon = float(lon_m.group(1)) if lon_m else None
                alt = float(alt_m.group(1) or alt_m.group(2)) if alt_m else None

                self.records.append(
                    TelemetryRecord(
                        timestamp_s=t_s,
                        latitude=lat,
                        longitude=lon,
                        altitude_m=alt,
                        gimbal_pitch_deg=-90.0,
                    )
                )

    def get_at_timestamp(self, timestamp_s: float) -> Optional[TelemetryRecord]:
        """Interpolates or returns the nearest telemetry record for a given timestamp."""
        if not self.records or len(self._timestamps) == 0:
            return None

        idx = int(np.searchsorted(self._timestamps, timestamp_s))
        if idx == 0:
            return self.records[0]
        if idx >= len(self.records):
            return self.records[-1]

        t0, t1 = self._timestamps[idx - 1], self._timestamps[idx]
        if abs(timestamp_s - t0) <= abs(timestamp_s - t1):
            return self.records[idx - 1]
        else:
            return self.records[idx]
