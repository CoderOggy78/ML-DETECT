"""
Open-Vocabulary Trajectory Anomaly Detector: Isolation Forest and Local Outlier Factor (LOF).
Discovers novel abnormal driving patterns without requiring predefined rule classifiers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from traffic_intelligence.discovery.embeddings import TrajectoryEmbeddingModel
from traffic_intelligence.discovery.features import TrajectoryFeatureExtractor
from traffic_intelligence.schema import SeverityLevel, TrafficEvent, Trajectory
from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.anomalies")


class TrajectoryAnomalyDetector:
    """Discovers outlier trajectories in high-dimensional feature space."""

    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.method = method
        self.contamination = contamination
        self.random_state = random_state
        self.embedding_model = TrajectoryEmbeddingModel(n_components=3, random_state=random_state)

    def detect_anomalies(self, trajectories: List[Trajectory]) -> Tuple[List[TrafficEvent], pd.DataFrame]:
        """Runs unsupervised anomaly detection on trajectory corpus."""
        if len(trajectories) < 4:
            return [], pd.DataFrame()

        # Extract features
        feat_df = TrajectoryFeatureExtractor.extract_feature_matrix(trajectories)
        numeric_cols = [c for c in feat_df.columns if c not in ["track_id", "class_name"]]
        X = feat_df[numeric_cols].values

        # Normalize & fit model
        scaled_X = self.embedding_model.scaler.fit_transform(X)

        if self.method == "lof":
            model = LocalOutlierFactor(n_neighbors=min(20, len(trajectories) - 1), contamination=self.contamination)
            preds = model.fit_predict(scaled_X)  # -1 for anomaly, 1 for normal
            scores = -model.negative_outlier_factor_
        else:
            model = IsolationForest(
                contamination=self.contamination, random_state=self.random_state, n_estimators=100
            )
            preds = model.fit_predict(scaled_X)
            scores = -model.score_samples(scaled_X)

        # 3D PCA embedding
        embeddings = self.embedding_model.fit_transform(X)
        feat_df["pca_1"] = embeddings[:, 0]
        feat_df["pca_2"] = embeddings[:, 1]
        feat_df["pca_3"] = embeddings[:, 2] if embeddings.shape[1] > 2 else 0.0
        feat_df["anomaly_score"] = scores
        feat_df["is_anomaly"] = preds == -1

        events: List[TrafficEvent] = []
        counter = 0

        for i, traj in enumerate(trajectories):
            if preds[i] == -1:  # Detected Anomaly
                counter += 1
                mid_pt = traj.points[len(traj.points) // 2]
                score = float(scores[i])

                sev = SeverityLevel.HIGH if score > np.percentile(scores, 95) else SeverityLevel.MEDIUM

                events.append(
                    TrafficEvent(
                        event_id=f"ANOMALY_{counter:05d}",
                        event_type="UNSUPERVISED_TRAJECTORY_ANOMALY",
                        start_timestamp_s=traj.start_timestamp_s,
                        end_timestamp_s=traj.end_timestamp_s,
                        start_frame=traj.start_frame,
                        end_frame=traj.end_frame,
                        involved_track_ids=[traj.track_id],
                        primary_class=traj.class_name,
                        severity=sev,
                        confidence=float(min(0.99, max(0.60, score))),
                        location_world=(mid_pt.world_x_m, mid_pt.world_y_m) if mid_pt.world_x_m else None,
                        location_pixel=(mid_pt.pixel_x, mid_pt.pixel_y),
                        description=f"Track #{traj.track_id} ({traj.class_name.value}) discovered as an open-vocabulary trajectory anomaly (score: {score:.3f}).",
                        metrics={
                            "anomaly_score": score,
                            "average_speed_kmh": traj.average_speed_kmh,
                            "total_distance_m": traj.total_distance_m,
                        },
                    )
                )

        logger.info(f"Discovered {len(events)} open-vocabulary anomaly trajectories.")
        return events, feat_df
