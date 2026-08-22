"""
Trajectory Embedding Model: Feature Standardization and Principal Component Analysis (PCA) Projection.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from traffic_intelligence.utils.logging import get_logger

logger = get_logger("traffic_intelligence.embeddings")


class TrajectoryEmbeddingModel:
    """Standardizes trajectory feature descriptors and projects them into low-dimensional latent space."""

    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.is_fitted = False

    def fit_transform(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Fits standardizer and PCA on feature matrix and returns low-dim embedding."""
        if len(feature_matrix) < self.n_components:
            # Fallback if too few samples
            self.scaler.fit(feature_matrix)
            self.is_fitted = True
            return feature_matrix[:, :self.n_components] if feature_matrix.shape[1] >= self.n_components else feature_matrix

        scaled = self.scaler.fit_transform(feature_matrix)
        embedding = self.pca.fit_transform(scaled)
        self.is_fitted = True
        logger.info(
            f"Fitted TrajectoryEmbeddingModel: explained variance ratio = {np.sum(self.pca.explained_variance_ratio_):.3f}"
        )
        return embedding

    def transform(self, feature_matrix: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return self.fit_transform(feature_matrix)
        scaled = self.scaler.transform(feature_matrix)
        return self.pca.transform(scaled)
