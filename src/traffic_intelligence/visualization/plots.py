"""
Analytics Plot Generator: Generates Plotly and Matplotlib analytical figures for speed, volume, and turning.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from traffic_intelligence.schema import Trajectory


class AnalyticsPlotGenerator:
    """Generates charts for traffic reports and dashboard views."""

    @staticmethod
    def plot_speed_distribution(trajectories: List[Trajectory]) -> go.Figure:
        speeds = []
        for t in trajectories:
            for p in t.points:
                if p.speed_kmh is not None:
                    speeds.append({"speed_kmh": p.speed_kmh, "class": t.class_name.value})

        if not speeds:
            fig = go.Figure()
            fig.update_layout(title="Speed Distribution (No Data)", template="plotly_dark")
            return fig

        df = pd.DataFrame(speeds)
        fig = px.histogram(
            df,
            x="speed_kmh",
            color="class",
            nbins=40,
            marginal="box",
            title="Class-Wise Speed Distribution (km/h)",
            template="plotly_dark",
            labels={"speed_kmh": "Speed (km/h)"},
        )
        return fig

    @staticmethod
    def plot_class_volume_pie(trajectories: List[Trajectory]) -> go.Figure:
        classes = [t.class_name.value for t in trajectories]
        if not classes:
            fig = go.Figure()
            fig.update_layout(title="Class Breakdown (No Data)", template="plotly_dark")
            return fig

        df = pd.DataFrame({"class": classes})
        counts = df["class"].value_counts().reset_index()
        counts.columns = ["class", "count"]

        fig = px.pie(
            counts,
            names="class",
            values="count",
            hole=0.4,
            title="Traffic Volume Composition by Road-User Class",
            template="plotly_dark",
        )
        return fig

    @staticmethod
    def plot_flow_over_time(flow_df: pd.DataFrame) -> go.Figure:
        if flow_df.empty:
            fig = go.Figure()
            fig.update_layout(title="Traffic Flow (No Data)", template="plotly_dark")
            return fig

        fig = px.line(
            flow_df,
            x="bin_start_s",
            y="flow_rate_veh_hr",
            title="Estimated Traffic Flow Rate Over Time (Vehicles / Hour)",
            labels={"bin_start_s": "Time (seconds)", "flow_rate_veh_hr": "Flow (veh/hr)"},
            template="plotly_dark",
            markers=True,
        )
        return fig

    @staticmethod
    def plot_anomaly_latent_space(feature_df: pd.DataFrame) -> go.Figure:
        if feature_df.empty or "pca_1" not in feature_df.columns:
            fig = go.Figure()
            fig.update_layout(title="Anomaly Latent Space (No Data)", template="plotly_dark")
            return fig

        fig = px.scatter_3d(
            feature_df,
            x="pca_1",
            y="pca_2",
            z="pca_3",
            color="is_anomaly",
            symbol="class_name",
            hover_name="track_id",
            title="Open-Vocabulary Trajectory Latent Space (PCA Projection)",
            template="plotly_dark",
            color_discrete_map={True: "#e74c3c", False: "#3498db"},
        )
        return fig
