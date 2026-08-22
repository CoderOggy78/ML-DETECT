"""
Trajectory Plotter: Generates static 2D vector plots and interactive Plotly trajectory maps.
"""

from __future__ import annotations

from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from traffic_intelligence.schema import RoadUserClass, Trajectory


class TrajectoryPlotter:
    """Renders 2D spatial trajectory flow diagrams and vector path charts."""

    CLASS_HEX_COLORS = {
        RoadUserClass.CAR: "#1f77b4",
        RoadUserClass.LGV: "#17becf",
        RoadUserClass.HGV: "#ff7f0e",
        RoadUserClass.BUS: "#bcbd22",
        RoadUserClass.TRUCK: "#2ca02c",
        RoadUserClass.MOTORCYCLE: "#9467bd",
        RoadUserClass.PEDESTRIAN: "#d62728",
        RoadUserClass.UNKNOWN: "#7f7f7f",
    }

    @classmethod
    def plot_trajectories_matplotlib(
        cls, trajectories: List[Trajectory], title: str = "Reconstructed Road-User Trajectories"
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.set_facecolor("#1e1e1e")
        fig.patch.set_facecolor("#121212")

        for t in trajectories:
            pts = t.get_world_coordinates_array()
            if len(pts) > 1:
                color = cls.CLASS_HEX_COLORS.get(t.class_name, "#ffffff")
                ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.7, linewidth=1.5)
                # Plot start point (dot) and end point (arrow)
                ax.scatter(pts[0, 0], pts[0, 1], color=color, s=20, edgecolors="white", linewidths=0.5)

        ax.set_title(title, color="white", fontsize=14, pad=12)
        ax.set_xlabel("World X (meters)", color="white")
        ax.set_ylabel("World Y (meters)", color="white")
        ax.tick_params(colors="white")
        ax.grid(True, linestyle="--", alpha=0.2, color="gray")
        plt.tight_layout()
        return fig

    @classmethod
    def plot_trajectories_plotly(cls, trajectories: List[Trajectory]) -> go.Figure:
        fig = go.Figure()

        for t in trajectories:
            pts = t.get_world_coordinates_array()
            if len(pts) > 1:
                color = cls.CLASS_HEX_COLORS.get(t.class_name, "#ffffff")
                hover_txt = f"ID: #{t.track_id}<br>Class: {t.class_name.value}<br>Avg Speed: {t.average_speed_kmh:.1f} km/h"

                fig.add_trace(
                    go.Scatter(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        mode="lines+markers",
                        marker=dict(size=3, color=color),
                        line=dict(color=color, width=2),
                        name=f"Track #{t.track_id} ({t.class_name.value})",
                        text=[hover_txt] * len(pts),
                        hoverinfo="text",
                        showlegend=False,
                    )
                )

        fig.update_layout(
            title="Interactive Trajectory Flow Map",
            template="plotly_dark",
            xaxis_title="World X (meters)",
            yaxis_title="World Y (meters)",
            height=600,
        )
        return fig
