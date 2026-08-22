"""
Streaming Video Reader and Frame Iterator with robust corruption handling and seeking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple, Union
import cv2
import numpy as np

from traffic_intelligence.data.metadata import MetadataReader
from traffic_intelligence.schema import VideoMetadata
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.video")


class VideoReader:
    """Stream-based video/image-sequence reader that avoids loading hours of footage into memory."""

    def __init__(
        self,
        source_path: Union[str, Path],
        frame_sample_rate: int = 1,
        max_frames: Optional[int] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ):
        self.source_path = Path(source_path)
        self.frame_sample_rate = max(1, frame_sample_rate)
        self.max_frames = max_frames
        self.target_size = target_size
        self.metadata: VideoMetadata = MetadataReader.inspect(self.source_path)

        self._is_dir = self.source_path.is_dir()
        self._image_files: List[Path] = []
        if self._is_dir:
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            self._image_files = sorted([p for p in self.source_path.iterdir() if p.suffix.lower() in image_extensions])

    def __iter__(self) -> Iterator[Tuple[int, float, np.ndarray]]:
        """Yields (frame_id, timestamp_s, frame_bgr)."""
        return self.stream_frames()

    def stream_frames(
        self, start_frame: int = 0, end_frame: Optional[int] = None
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Streams frames lazily one by one or in sampled stride."""
        if self._is_dir:
            yield from self._stream_image_dir(start_frame, end_frame)
        else:
            yield from self._stream_video_file(start_frame, end_frame)

    def _stream_image_dir(
        self, start_frame: int, end_frame: Optional[int]
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        total = len(self._image_files)
        stop = min(end_frame or total, total)
        if self.max_frames:
            stop = min(stop, start_frame + self.max_frames * self.frame_sample_rate)

        count = 0
        for idx in range(start_frame, stop, self.frame_sample_rate):
            img_path = self._image_files[idx]
            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.warning(f"Corrupted or unreadable image frame at {img_path}. Skipping.")
                continue

            if self.target_size:
                frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

            timestamp_s = idx / self.metadata.fps
            yield (idx, timestamp_s, frame)
            count += 1
            if self.max_frames and count >= self.max_frames:
                break

    def _stream_video_file(
        self, start_frame: int, end_frame: Optional[int]
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        cap = cv2.VideoCapture(str(self.source_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {self.source_path}")

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame_id = start_frame
        stop = end_frame if end_frame is not None else float("inf")
        yielded_count = 0

        try:
            while cap.isOpened() and current_frame_id <= stop:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if (current_frame_id - start_frame) % self.frame_sample_rate == 0:
                    if self.target_size:
                        frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

                    timestamp_s = current_frame_id / self.metadata.fps
                    yield (current_frame_id, timestamp_s, frame)
                    yielded_count += 1

                    if self.max_frames and yielded_count >= self.max_frames:
                        break

                current_frame_id += 1
        finally:
            cap.release()

    def stream_chunks(
        self, chunk_size: int = 300
    ) -> Generator[List[Tuple[int, float, np.ndarray]], None, None]:
        """Yields chunks of frames to enable streaming batch inference and checkpointing."""
        chunk: List[Tuple[int, float, np.ndarray]] = []
        for item in self.stream_frames():
            chunk.append(item)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


FrameIterator = VideoReader
