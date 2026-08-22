"""
Heatmap Plotter: Density, Speed, and Conflict 2D Heatmaps in Matplotlib and Plotly.
"""

from __future__ import annotations

from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


class HeatmapPlotter:
    """Renders 2D spatial heatmap rasters."""

    @staticmethod
    def plot_heatmap_matplotlib(
        heatmap_matrix: np.ndarray,
        bounds: Tuple[float, float, float, float],
        title: str = "Spatial Heatmap",
        cmap: str = "hot",
    ) -> plt.Figure:
        min_x, max_x, min_y, max_y = bounds
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        ax.set_facecolor("#1e1e1e")
        fig.patch.set_facecolor("#121212")

        im = ax.imshow(
            heatmap_matrix,
            origin="lower",
            extent=[min_x, max_x, min_y, max_y],
            cmap=cmap,
            aspect="auto",
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        ax.set_title(title, color="white", fontsize=13, pad=10)
        ax.set_xlabel("World X (meters)", color="white")
        ax.set_ylabel("World Y (meters)", color="white")
        ax.tick_params(colors="white")
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_heatmap_plotly(
        heatmap_matrix: np.ndarray,
        bounds: Tuple[float, float, float, float],
        title: str = "Interactive Spatial Heatmap",
        colorscale: str = "Hot",
    ) -> go.Figure:
        min_x, max_x, min_y, max_y = bounds
        rows, cols = heatmap_matrix.shape
        x_coords = np.linspace(min_x, max_x, cols)
        y_coords = np.linspace(min_y, max_y, rows)

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_matrix,
                x=x_coords,
                y=y_coords,
                colorscale=colorscale,
            )
        )
        fig.update_layout(
            title=title,
            template="plotly_dark",
            xaxis_title="World X (meters)",
            yaxis_title="World Y (meters)",
            height=550,
        )
        return fig
