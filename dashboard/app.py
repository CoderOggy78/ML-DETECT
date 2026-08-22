"""
Streamlit Interactive Dashboard for Aerial Traffic Intelligence & Trajectory Discovery.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Aerial Traffic Intelligence Platform",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #38bdf8; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.0rem; color: #94a3b8; margin-bottom: 1.5rem; }
    .kpi-container { background: #1e293b; border-radius: 8px; padding: 15px; border: 1px solid #334155; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("🚁 Navigation & Data")

# Select run output directory
default_runs = sorted(list(Path("outputs").glob("*")))
run_options = [str(p) for p in default_runs if p.is_dir()] or ["outputs/run_default"]

selected_run_dir = st.sidebar.selectbox("Select Output Run Directory:", run_options, index=0)
run_path = Path(selected_run_dir)

# Load data helper functions
@st.cache_data
def load_run_data(dir_path: str):
    p = Path(dir_path)
    summary = {}
    trajs_df = pd.DataFrame()
    conflicts_df = pd.DataFrame()
    events = []
    queues_df = pd.DataFrame()
    speeds_df = pd.DataFrame()
    turn_df = pd.DataFrame()

    sum_file = p / "analytics" / "traffic_summary.json"
    if sum_file.exists():
        with open(sum_file, "r") as f:
            summary = json.load(f)

    traj_file = p / "trajectories" / "trajectories.parquet"
    if not traj_file.exists():
        traj_file = p / "trajectories" / "trajectories.csv"
    if traj_file.exists():
        trajs_df = pd.read_parquet(traj_file) if traj_file.suffix == ".parquet" else pd.read_csv(traj_file)

    conf_file = p / "analytics" / "conflicts.csv"
    if conf_file.exists():
        conflicts_df = pd.read_csv(conf_file)

    ev_file = p / "events" / "events.json"
    if ev_file.exists():
        with open(ev_file, "r") as f:
            events = json.load(f)

    q_file = p / "analytics" / "queue_statistics.csv"
    if q_file.exists():
        queues_df = pd.read_csv(q_file)

    spd_file = p / "analytics" / "speed_statistics.csv"
    if spd_file.exists():
        speeds_df = pd.read_csv(spd_file)

    t_file = p / "analytics" / "turning_movements.csv"
    if t_file.exists():
        turn_df = pd.read_csv(t_file)

    return summary, trajs_df, conflicts_df, events, queues_df, speeds_df, turn_df


summary, trajs_df, conflicts_df, events, queues_df, speeds_df, turn_df = load_run_data(str(run_path))

# Title
st.markdown('<div class="main-header">🚁 AERIAL TRAFFIC INTELLIGENCE PLATFORM</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Zero-Annotation Trajectory Reconstruction, Surrogate Safety Conflicts, and Open-Vocabulary Discovery</div>',
    unsafe_allow_html=True,
)

# Top KPIs
meta = summary.get("metadata", {})
flow_meta = summary.get("flow_and_congestion", {})
safety_meta = summary.get("surrogate_safety", {})
qual_meta = summary.get("data_quality_and_integrity", {})

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Observed Vehicles", meta.get("total_vehicles_observed", len(trajs_df["track_id"].unique()) if not trajs_df.empty else 0))
col2.metric("Hourly Volume", f"{flow_meta.get('estimated_hourly_volume', 0)} veh/h")
col3.metric("85th % Speed", f"{flow_meta.get('p85_speed_kmh', 0.0):.1f} km/h")
col4.metric("Congestion Level", flow_meta.get("congestion_level", "FREE_FLOW"))
col5.metric("Conflicts (TTC/DRAC)", safety_meta.get("total_conflicts", len(conflicts_df)))
col6.metric("Data Quality Grade", qual_meta.get("data_confidence_grade", "GOOD"))

st.markdown("---")

# Main Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "🗺️ Trajectory Flows",
        "🔥 Spatial Heatmaps",
        "⚠️ Surrogate Safety Conflicts",
        "🚦 Turning Movements & Queues",
        "🔍 Open-Vocabulary Anomalies",
        "📊 Data Quality & Camera Motion",
    ]
)

# Tab 1: Trajectory Flows
with tab1:
    st.subheader("Interactive 2D Trajectory Vector Map")
    if not trajs_df.empty:
        # Class filter
        avail_classes = sorted(trajs_df["class_name"].unique())
        selected_classes = st.multiselect("Filter Road-User Classes:", avail_classes, default=avail_classes)
        filtered_df = trajs_df[trajs_df["class_name"].isin(selected_classes)]

        fig_traj = px.line(
            filtered_df,
            x="world_x_m",
            y="world_y_m",
            color="class_name",
            line_group="track_id",
            hover_data=["track_id", "speed_kmh", "movement_type"],
            title="Reconstructed Multi-Agent Trajectories (World Coordinates)",
            template="plotly_dark",
        )
        fig_traj.update_layout(height=650)
        st.plotly_chart(fig_traj, use_container_width=True)

        st.subheader("Class-Wise Speed & Kinematic Statistics")
        if not speeds_df.empty:
            st.dataframe(speeds_df, use_container_width=True)
    else:
        st.info("No trajectory data available in the selected run.")

# Tab 2: Spatial Heatmaps
with tab2:
    st.subheader("2D Spatial Density and Speed Heatmaps")
    if not trajs_df.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig_dens = px.density_heatmap(
                trajs_df,
                x="world_x_m",
                y="world_y_m",
                nbinsx=50,
                nbinsy=50,
                color_continuous_scale="Viridis",
                title="Trajectory Passage Density Heatmap",
                template="plotly_dark",
            )
            fig_dens.update_layout(height=480)
            st.plotly_chart(fig_dens, use_container_width=True)

        with c2:
            fig_spd_heat = px.density_heatmap(
                trajs_df,
                x="world_x_m",
                y="world_y_m",
                z="speed_kmh",
                histfunc="avg",
                nbinsx=50,
                nbinsy=50,
                color_continuous_scale="Jet",
                title="Spatial Average Speed Heatmap (km/h)",
                template="plotly_dark",
            )
            fig_spd_heat.update_layout(height=480)
            st.plotly_chart(fig_spd_heat, use_container_width=True)

# Tab 3: Surrogate Safety Conflicts
with tab3:
    st.subheader("Surrogate Safety Conflicts (TTC, PET, DRAC)")
    if not conflicts_df.empty:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_conf = px.scatter(
                conflicts_df,
                x="world_x_m",
                y="world_y_m",
                color="severity",
                symbol="conflict_type",
                hover_data=["event_id", "primary_track_id", "secondary_track_id", "ttc_s", "drac_mps2"],
                color_discrete_map={"CRITICAL": "#ef4444", "HIGH": "#f97316", "MEDIUM": "#eab308", "LOW": "#22c55e"},
                title="Conflict Spatial Hotspots",
                template="plotly_dark",
            )
            fig_conf.update_traces(marker=dict(size=14, line=dict(width=1, color="white")))
            fig_conf.update_layout(height=520)
            st.plotly_chart(fig_conf, use_container_width=True)

        with c2:
            st.markdown("#### Conflict Severity Breakdown")
            sev_counts = conflicts_df["severity"].value_counts().reset_index()
            sev_counts.columns = ["Severity", "Count"]
            fig_pie = px.pie(sev_counts, names="Severity", values="Count", hole=0.4, template="plotly_dark")
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### Detailed Conflict Event Log")
        st.dataframe(conflicts_df, use_container_width=True)
    else:
        st.info("No safety conflicts detected.")

# Tab 4: Turning Movements & Queues
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Intersection Turning Movement Matrix")
        if not turn_df.empty:
            st.dataframe(turn_df, use_container_width=True)
            fig_turn = px.bar(
                turn_df,
                x="movement_type",
                y="count",
                color="origin_zone",
                title="Turning Volumes by Movement Type",
                template="plotly_dark",
            )
            st.plotly_chart(fig_turn, use_container_width=True)
        else:
            st.info("No turning movement matrix data.")

    with c2:
        st.subheader("Queue Evolution & Dynamics")
        if not queues_df.empty:
            st.dataframe(queues_df, use_container_width=True)
            fig_q = px.line(
                queues_df,
                x="timestamp_s",
                y="queue_length_m",
                color="queue_id",
                title="Queue Length Evolution (meters)",
                template="plotly_dark",
                markers=True,
            )
            st.plotly_chart(fig_q, use_container_width=True)
        else:
            st.info("No active queues formed in this video segment.")

# Tab 5: Open-Vocabulary Anomalies
with tab5:
    st.subheader("Open-Vocabulary Trajectory Anomaly Discovery (Isolation Forest & PCA Latent Space)")
    if events:
        anomaly_list = [e for e in events if "ANOMALY" in e.get("event_type", "")]
        st.write(f"Discovered **{len(anomaly_list)}** open-vocabulary behavioral anomalies without predefined rule labels.")
        st.json(anomaly_list[:5])
    else:
        st.info("No anomaly event data.")

# Tab 6: Data Quality & Camera Motion
with tab6:
    st.subheader("Unsupervised Data Quality & Camera Motion Residuals")
    st.json(qual_meta)
