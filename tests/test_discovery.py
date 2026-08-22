"""
Unit tests for Open-Vocabulary Trajectory Discovery, Feature Extraction, and Anomaly Detection.
"""

import numpy as np
import pytest
from traffic_intelligence.discovery.anomalies import TrajectoryAnomalyDetector
from traffic_intelligence.discovery.clustering import TrajectoryClusterEngine
from traffic_intelligence.discovery.embeddings import TrajectoryEmbeddingModel
from traffic_intelligence.discovery.features import TrajectoryFeatureExtractor


def test_trajectory_feature_extractor(sample_trajectories):
    feat_df = TrajectoryFeatureExtractor.extract_feature_matrix(sample_trajectories)
    assert len(feat_df) == len(sample_trajectories)
    assert "mean_speed_mps" in feat_df.columns
    assert "tortuosity_ratio" in feat_df.columns
    assert "accel_skewness" in feat_df.columns
    assert not feat_df.isna().any().any()


def test_trajectory_embedding_model(sample_trajectories):
    feat_df = TrajectoryFeatureExtractor.extract_feature_matrix(sample_trajectories)
    num_cols = [c for c in feat_df.columns if c not in ["track_id", "class_name"]]
    X = feat_df[num_cols].values

    emb_model = TrajectoryEmbeddingModel(n_components=2)
    embs = emb_model.fit_transform(X)
    assert embs.shape == (len(sample_trajectories), 2)


def test_anomaly_detector_and_clustering(sample_trajectories):
    detector = TrajectoryAnomalyDetector(contamination=0.15)
    events, df = detector.detect_anomalies(sample_trajectories)
    assert isinstance(events, list)
    assert not df.empty
    assert "anomaly_score" in df.columns

    clusterer = TrajectoryClusterEngine(method="dbscan", eps=2.0)
    labels, clusters = clusterer.cluster_embeddings(df[["pca_1", "pca_2"]].values, sample_trajectories)
    assert len(labels) == len(sample_trajectories)
