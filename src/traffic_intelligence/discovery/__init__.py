"""
Open-Vocabulary Trajectory Discovery & Anomaly Engine:
Unsupervised feature extraction, embedding projection, density clustering, and anomaly scoring.
"""

from traffic_intelligence.discovery.features import TrajectoryFeatureExtractor
from traffic_intelligence.discovery.embeddings import TrajectoryEmbeddingModel
from traffic_intelligence.discovery.clustering import TrajectoryClusterEngine, TrajectoryCluster
from traffic_intelligence.discovery.anomalies import TrajectoryAnomalyDetector

__all__ = [
    "TrajectoryFeatureExtractor",
    "TrajectoryEmbeddingModel",
    "TrajectoryClusterEngine",
    "TrajectoryCluster",
    "TrajectoryAnomalyDetector",
]
