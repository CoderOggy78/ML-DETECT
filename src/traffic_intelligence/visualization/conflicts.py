"""
Conflict Plotter: Visualizes surrogate safety conflict hotspots, TTC distributions, and interaction events.
"""

from __future__ import annotations

from typing import List
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from traffic_intelligence.schema import ConflictEvent


class ConflictPlotter:
    """Renders conflict scatter maps and surrogate safety charts."""

    @staticmethod
    def plot_conflict_scatter_plotly(conflicts: List[ConflictEvent]) -> go.Figure:
        if not conflicts:
            fig = go.Figure()
            fig.update_layout(title="No Conflicts Detected", template="plotly_dark")
            return fig

        x_vals = [c.location_world[0] if c.location_world else c.location_pixel[0] for c in conflicts]
        y_vals = [c.location_world[1] if c.location_world else c.location_pixel[1] for c in conflicts]
        types = [c.conflict_type for c in conflicts]
        severities = [c.severity.value for c in conflicts]
        ttcs = [c.ttc_s or 0.0 for c in conflicts]
        hover = [
            f"Event: {c.event_id}<br>Type: {c.conflict_type}<br>Severity: {c.severity.value}<br>TTC: {c.ttc_s or 'N/A'}s<br>DRAC: {c.drac_mps2 or 'N/A'} m/s²"
            for c in conflicts
        ]

        fig = px.scatter(
            x=x_vals,
            y=y_vals,
            color=severities,
            symbol=types,
            hover_name=types,
            color_discrete_map={"CRITICAL": "#e74c3c", "HIGH": "#e67e22", "MEDIUM": "#f1c40f", "LOW": "#2ecc71"},
            title="Surrogate Safety Conflict Hotspots (World Coordinates)",
            labels={"x": "World X (meters)", "y": "World Y (meters)"},
            template="plotly_dark",
        )
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")), hovertext=hover)
        return fig
