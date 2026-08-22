"""
Base Tracker interface and linear assignment helpers (Hungarian / Jonker-Volgenant matching).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from traffic_intelligence.schema import Detection, Track


def linear_assignment(cost_matrix: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves optimal bipartite matching using the Hungarian algorithm (scipy linear_sum_assignment).
    Returns (matches, unmatched_a, unmatched_b).
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_a = list(range(cost_matrix.shape[0]))
    unmatched_b = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= threshold:
            matches.append([r, c])
            if r in unmatched_a:
                unmatched_a.remove(r)
            if c in unmatched_b:
                unmatched_b.remove(c)

    return (
        np.array(matches, dtype=int) if matches else np.empty((0, 2), dtype=int),
        np.array(unmatched_a, dtype=int),
        np.array(unmatched_b, dtype=int),
    )


def compute_iou_cost_matrix(
    boxes_a: List[Tuple[float, float, float, float]], boxes_b: List[Tuple[float, float, float, float]]
) -> np.ndarray:
    """Computes (1.0 - IoU) cost matrix between two sets of bounding boxes."""
    if not boxes_a or not boxes_b:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float64)

    matrix = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            x1 = max(a[0], b[0])
            y1 = max(a[1], b[1])
            x2 = min(a[2], b[2])
            y2 = min(a[3], b[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            union = area_a + area_b - inter
            iou = inter / union if union > 0 else 0.0
            matrix[i, j] = 1.0 - iou
    return matrix


class BaseTracker(ABC):
    """Abstract interface for multi-object trackers."""

    @abstractmethod
    def update(
        self,
        detections: List[Detection],
        frame_bgr: Optional[np.ndarray] = None,
        cmc_matrix: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """Updates tracker state with new frame detections and returns active tracks."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets tracker state."""
        pass
