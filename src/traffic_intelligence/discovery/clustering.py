"""
Trajectory Clustering Engine: groups similar patterns of movement into representative behavioral archetypes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from traffic_intelligence.schema import Trajectory
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.clustering")


@dataclass
class TrajectoryCluster:
    cluster_id: int
    size: int
    representative_track_id: int
    mean_speed_kmh: float
    mean_duration_s: float
    description: str


class TrajectoryClusterEngine:
    """Discovers natural behavioral clusters among road users using unsupervised clustering."""

    def __init__(self, method: str = "dbscan", eps: float = 1.5, min_samples: int = 2, n_clusters: int = 5):
        self.method = method
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters = n_clusters

    def cluster_embeddings(
        self, embeddings: np.ndarray, trajectories: List[Trajectory]
    ) -> Tuple[np.ndarray, List[TrajectoryCluster]]:
        """Clusters low-dimensional embeddings and identifies archetype centroids."""
        N = len(embeddings)
        if N < 2:
            labels = np.zeros(N, dtype=int)
            return labels, []

        if self.method == "kmeans":
            k = min(self.n_clusters, N)
            model = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = model.fit_predict(embeddings)
        else:
            model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = model.fit_predict(embeddings)

        clusters: List[TrajectoryCluster] = []
        unique_labels = sorted(set(labels))

        for lbl in unique_labels:
            mask = labels == lbl
            cluster_trajs = [trajectories[i] for i in range(N) if mask[i]]
            cluster_embs = embeddings[mask]

            if not cluster_trajs:
                continue

            # Find representative closest to cluster center
            centroid = np.mean(cluster_embs, axis=0)
            dists = np.linalg.norm(cluster_embs - centroid, axis=1)
            best_idx = int(np.argmin(dists))
            rep_track = cluster_trajs[best_idx].track_id

            mean_spd = float(np.mean([t.average_speed_kmh for t in cluster_trajs]))
            mean_dur = float(np.mean([t.duration_s for t in cluster_trajs]))
            desc = "Outlier / Unique Pattern" if lbl == -1 else f"Behavioral Archetype #{lbl}"

            clusters.append(
                TrajectoryCluster(
                    cluster_id=int(lbl),
                    size=len(cluster_trajs),
                    representative_track_id=rep_track,
                    mean_speed_kmh=mean_spd,
                    mean_duration_s=mean_dur,
                    description=desc,
                )
            )

        logger.info(f"Discovered {len(clusters)} distinct trajectory behavioral clusters.")
        return labels, clusters
