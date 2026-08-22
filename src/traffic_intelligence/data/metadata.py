"""
Video and media metadata extraction and auto-inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import cv2

from traffic_intelligence.schema import VideoMetadata
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.metadata")


class MetadataReader:
    """Inspects video or image sequence sources to extract technical metadata."""

    @staticmethod
    def inspect(source_path: Union[str, Path], telemetry_path: Optional[Union[str, Path]] = None) -> VideoMetadata:
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Media source does not exist: {path.resolve()}")

        # Case 1: Directory of image sequences
        if path.is_dir():
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            image_files = sorted([p for p in path.iterdir() if p.suffix.lower() in image_extensions])
            if not image_files:
                raise ValueError(f"No valid image files found in directory: {path}")

            sample_img = cv2.imread(str(image_files[0]))
            if sample_img is None:
                raise ValueError(f"Could not read first image frame: {image_files[0]}")

            h, w = sample_img.shape[:2]
            total_frames = len(image_files)
            fps = 30.0  # Default assumption for image sequence if not specified
            duration_s = total_frames / fps
            codec = "IMAGE_SEQUENCE"

        # Case 2: Video file
        else:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file with OpenCV: {path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps is None or fps > 240:
                fps = 30.0  # Fallback for variable/corrupted header FPS

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()
            cap.release()

            if total_frames <= 0:
                total_frames = 1
            duration_s = total_frames / fps

        has_telemetry = False
        telemetry_samples = 0
        if telemetry_path and Path(telemetry_path).exists():
            has_telemetry = True

        metadata = VideoMetadata(
            source_path=str(path.resolve()),
            fps=float(fps),
            width=w,
            height=h,
            total_frames=total_frames,
            duration_s=float(duration_s),
            codec=codec if codec else "UNKNOWN",
            has_telemetry=has_telemetry,
            telemetry_samples_count=telemetry_samples,
            estimated_gsd_m=0.05,  # 5cm/px default
        )
        logger.info(
            f"Inspected media: {path.name} | {w}x{h} @ {fps:.2f}fps | {total_frames} frames ({duration_s:.1f}s)"
        )
        return metadata
