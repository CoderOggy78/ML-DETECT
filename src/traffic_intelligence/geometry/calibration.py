"""
Camera calibration manager supporting Ground Plane Homography, Telemetry, and Scale Factors.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from traffic_intelligence.geometry.coordinates import PixelToWorldTransformer
from traffic_intelligence.geometry.homography import HomographyEstimator
from traffic_intelligence.schema import TelemetryRecord
from traffic_intelligence.utils.config import load_config
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.calibration")


class CalibrationManager:
    """Configures and builds PixelToWorld transformers from multiple calibration sources."""

    def __init__(self, config_or_path: Optional[Union[Dict[str, Any], str, Path]] = None):
        self.config: Dict[str, Any] = {}
        if isinstance(config_or_path, (str, Path)):
            p = Path(config_or_path)
            if p.exists():
                self.config = load_config(p).get("calibration", {})
        elif isinstance(config_or_path, dict):
            self.config = config_or_path.get("calibration", config_or_path)

        self.transformer: Optional[PixelToWorldTransformer] = None
        self._build_transformer()

    def _build_transformer(self) -> None:
        mode = self.config.get("mode", "auto")

        # 1. Homography points provided
        hom_cfg = self.config.get("homography", {})
        img_pts = hom_cfg.get("image_points", [])
        wld_pts = hom_cfg.get("world_points", [])
        mat = hom_cfg.get("matrix", None)

        if mat is not None:
            H = np.array(mat, dtype=np.float64)
            self.transformer = PixelToWorldTransformer(homography_matrix=H)
            logger.info("Initialized PixelToWorldTransformer with explicit 3x3 homography matrix.")
            return

        if len(img_pts) >= 4 and len(wld_pts) >= 4:
            H, _ = HomographyEstimator.estimate_from_points(np.array(img_pts), np.array(wld_pts))
            if H is not None:
                self.transformer = PixelToWorldTransformer(homography_matrix=H)
                logger.info(f"Computed ground-plane Homography from {len(img_pts)} reference points.")
                return

        # 2. Scale factor (GSD)
        scale_cfg = self.config.get("scale", {})
        mpp = scale_cfg.get("meters_per_pixel", None)
        origin = tuple(scale_cfg.get("origin_image_coord", [0.0, 0.0]))

        if mpp is not None and mpp > 0:
            self.transformer = PixelToWorldTransformer(meters_per_pixel=mpp, origin_pixel=origin)
            logger.info(f"Initialized PixelToWorldTransformer with scale factor {mpp:.4f} m/px.")
            return

        # Fallback default: 0.05 meters per pixel (5 cm/px typical for 50-80m drone altitude)
        self.transformer = PixelToWorldTransformer(meters_per_pixel=0.05, origin_pixel=(0.0, 0.0))
        logger.info("Initialized fallback PixelToWorldTransformer with default 0.05 m/px.")

    def get_transformer(self, telemetry: Optional[TelemetryRecord] = None) -> PixelToWorldTransformer:
        """Returns transformer, dynamically adapting to telemetry altitude/gimbal if configured."""
        if telemetry is not None and self.config.get("mode") == "telemetry" and telemetry.altitude_m:
            # Estimate meters_per_pixel dynamically from altitude and sensor specs
            # GSD = (altitude * sensor_width) / (focal_length * image_width)
            tel_cfg = self.config.get("telemetry", {}).get("camera", {})
            focal_mm = tel_cfg.get("focal_length_mm", 24.0)
            sensor_w_mm = tel_cfg.get("sensor_width_mm", 13.2)
            
            # Assuming ~4K standard frame width if not known: 3840 or 1920
            # For 24mm on 13.2mm sensor at 60m height: GSD ~ (60 * 13.2)/(24 * 3840) ~ 0.0086 m/px
            gsd = (telemetry.altitude_m * (sensor_w_mm / 1000.0)) / ((focal_mm / 1000.0) * 1920)
            return PixelToWorldTransformer(meters_per_pixel=max(0.01, gsd))

        return self.transformer or PixelToWorldTransformer(meters_per_pixel=0.05)
