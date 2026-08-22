"""
Dataset adapter and multi-video discovery engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from traffic_intelligence.data.metadata import MetadataReader
from traffic_intelligence.schema import VideoMetadata
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.dataset")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}
TELEMETRY_EXTENSIONS = {".csv", ".json", ".srt", ".gpx"}


@dataclass
class VideoEntry:
    video_path: Path
    telemetry_path: Optional[Path]
    metadata: VideoMetadata


class DatasetAdapter:
    """Discovers and organizes arbitrary video inputs, paired telemetry, and directory structures."""

    def __init__(self, input_path: Union[str, Path]):
        self.input_path = Path(input_path)
        self.entries: List[VideoEntry] = []
        self._discover()

    def _discover(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {self.input_path.resolve()}")

        # Single file
        if self.input_path.is_file():
            if self.input_path.suffix.lower() in VIDEO_EXTENSIONS:
                telemetry = self._find_matching_telemetry(self.input_path)
                meta = MetadataReader.inspect(self.input_path, telemetry)
                self.entries.append(VideoEntry(self.input_path, telemetry, meta))
            else:
                raise ValueError(f"Input file is not a supported video format: {self.input_path}")

        # Directory
        elif self.input_path.is_dir():
            # Check if this directory is an image sequence
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            has_images = any(p.suffix.lower() in image_extensions for p in self.input_path.iterdir() if p.is_file())
            has_videos = any(p.suffix.lower() in VIDEO_EXTENSIONS for p in self.input_path.iterdir() if p.is_file())

            if has_images and not has_videos:
                # Directory is a single image sequence
                telemetry = self._find_matching_telemetry(self.input_path)
                meta = MetadataReader.inspect(self.input_path, telemetry)
                self.entries.append(VideoEntry(self.input_path, telemetry, meta))
            else:
                # Search recursively for video files
                video_files = sorted(
                    [p for p in self.input_path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
                )
                for vf in video_files:
                    telemetry = self._find_matching_telemetry(vf)
                    meta = MetadataReader.inspect(vf, telemetry)
                    self.entries.append(VideoEntry(vf, telemetry, meta))

        logger.info(f"Discovered {len(self.entries)} video/stream sources in {self.input_path}")

    def _find_matching_telemetry(self, video_path: Path) -> Optional[Path]:
        stem = video_path.stem
        parent = video_path.parent
        
        # Check identical stem with telemetry extensions
        for ext in TELEMETRY_EXTENSIONS:
            candidate = parent / f"{stem}{ext}"
            if candidate.exists():
                return candidate
            # Check telemetry subfolder
            candidate_sub = parent / "telemetry" / f"{stem}{ext}"
            if candidate_sub.exists():
                return candidate_sub

        return None
