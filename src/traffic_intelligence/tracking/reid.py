"""
Visual Appearance Re-Identification (ReID) extraction and Cosine Distance matching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.reid")


def cosine_distance_matrix(features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
    """
    Computes pairwise cosine distance: 1.0 - (u . v) / (||u|| * ||v||).
    Matrix shape: (len(features_a), len(features_b)).
    """
    if len(features_a) == 0 or len(features_b) == 0:
        return np.empty((len(features_a), len(features_b)), dtype=np.float64)

    norm_a = np.linalg.norm(features_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(features_b, axis=1, keepdims=True)
    norm_a = np.where(norm_a > 1e-7, norm_a, 1.0)
    norm_b = np.where(norm_b > 1e-7, norm_b, 1.0)

    unit_a = features_a / norm_a
    unit_b = features_b / norm_b

    similarity = unit_a @ unit_b.T
    return np.clip(1.0 - similarity, 0.0, 2.0)


class BaseReID(ABC):
    """Abstract interface for extracting appearance feature vectors from bounding box crops."""

    @abstractmethod
    def extract_features(self, frame_bgr: np.ndarray, bboxes_xyxy: List[Tuple[float, float, float, float]]) -> np.ndarray:
        pass


class AppearanceExtractor(BaseReID):
    """
    Lightweight, fast visual feature extractor combining multi-channel HSV/LAB color histograms
    and spatial gradient distributions. Works in zero-shot mode without requiring external model weights.
    """

    def __init__(self, feature_dim: int = 128):
        self.feature_dim = feature_dim

    def extract_features(
        self, frame_bgr: np.ndarray, bboxes_xyxy: List[Tuple[float, float, float, float]]
    ) -> np.ndarray:
        if not bboxes_xyxy:
            return np.empty((0, self.feature_dim), dtype=np.float64)

        features = []
        h_img, w_img = frame_bgr.shape[:2]

        for box in bboxes_xyxy:
            x1 = max(0, min(int(box[0]), w_img - 1))
            y1 = max(0, min(int(box[1]), h_img - 1))
            x2 = max(x1 + 1, min(int(box[2]), w_img))
            y2 = max(y1 + 1, min(int(box[3]), h_img))

            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                features.append(np.zeros(self.feature_dim, dtype=np.float64))
                continue

            # Standardize crop size
            resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)

            # Compute normalized color histograms
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
            hist_l = cv2.calcHist([lab], [0], None, [32], [0, 256]).flatten()
            hist_b = cv2.calcHist([lab], [2], None, [32], [0, 256]).flatten()

            vec = np.concatenate([hist_h, hist_s, hist_l, hist_b])
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec /= norm

            features.append(vec)

        return np.array(features, dtype=np.float64)
