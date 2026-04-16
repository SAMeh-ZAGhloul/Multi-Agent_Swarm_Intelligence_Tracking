"""Analytics - Historical performance and statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.theme import apply_theme

apply_theme()

st.set_page_config(page_title="Analytics", layout="wide")

st.title("📊 Analytics Dashboard")
st.markdown("Historical performance metrics and system statistics.")

# Generate demo analytics data
def generate_analytics_data() -> dict:
    """Generate demo analytics data."""
    np.random.seed(42)

    # Time series data (last 24 hours)
    hours = np.arange(24)

    # Track count over time
    track_counts = 10 + 15 * np.sin(hours * np.pi / 12) + np.random.randn(24) * 3
    track_counts = np.clip(track_counts, 0, 50).astype(int)

    # Threat distribution
    threat_dist = {
        "LOW": 45,
        "MEDIUM": 35,
        "HIGH": 15,
        "CRITICAL": 5,
    }

    # Behavior distribution
    behavior_dist = {
        "TRANSIT": 30,
        "ATTACK": 15,
        "SCATTER": 20,
        "ENCIRCLE": 10,
        "DECOY": 15,
        "UNKNOWN": 10,
    }

    # Response actions
    response_actions = {
        "INTERCEPT": 12,
        "JAM": 8,
        "ESCALATE": 5,
        "IGNORE": 25,
    }

    # Performance metrics
    performance = {
        "ekf_cycle_time_ms": 0.8,
        "hungarian_time_ms": 4.2,
        "gnn_inference_ms": 18.5,
        "end_to_end_latency_ms": 85.0,
        "track_continuity_rate": 0.96,
    }

    return {
        "hours": hours,
        "track_counts": track_counts,
        "threat_dist": threat_dist,
        "behavior_dist": behavior_dist,
        "response_actions": response_actions,
        "performance": performance,
    }


data = generate_analytics_data()

# Main layout
tab1, tab2, tab3 = st.tabs(["Overview", "Performance", "Threat Analysis"])

with tab1:
    st.subheader("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Tracks (24h)",
            f"{sum(data['track_counts']):,}",
            delta="+12% from previous",
        )

    with col2:
        st.metric(
            "Active Swarms",
            "3",
            delta=None,
        )

    with col3:
        st.metric(
            "Avg Threat Score",
            "0.45",
            delta="-0.05",
        )

    with col4:
        st.metric(
            "Response Actions",
            "50",
            delta="+8",
        )

    # Track count over time
    st.markdown("---")
    st.subheader("Track Count Over Time")

    fig_tracks = go.Figure(
        data=[
            go.Scatter(
                x=data["hours"],
                y=data["track_counts"],
                mode="lines+markers",
                line={"color": "#1f77b4", "width": 2},
                fill="tozeroy",
                fillcolor="rgba(31, 119, 180, 0.2)",
                name="Tracks",
            )
        ]
    )

    fig_tracks.update_layout(
        xaxis={"title": "Hour", "range": [0, 23]},
        yaxis={"title": "Number of Tracks"},
        height=400,
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )

    st.plotly_chart(fig_tracks, use_container_width=True)

    # Behavior distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Behavior Distribution")

        fig_behavior = go.Figure(
            data=[
                go.Bar(
                    x=list(data["behavior_dist"].keys()),
                    y=list(data["behavior_dist"].values()),
                    marker_color=[
                        "#1f77b4",
                        "#dc3545",
                        "#fd7e14",
                        "#9467bd",
                        "#7f7f7f",
                        "#d3d3d3",
                    ],
                    text=[str(v) for v in data["behavior_dist"].values()],
                    textposition="auto",
                )
            ]
        )

        fig_behavior.update_layout(
            xaxis={"title": "Behavior Class"},
            yaxis={"title": "Count"},
            height=400,
            showlegend=False,
            margin={"l": 0, "r": 0, "t": 50, "b": 0},
        )

        st.plotly_chart(fig_behavior, use_container_width=True)

    with col2:
        st.subheader("Response Actions")

        fig_response = go.Figure(
            data=[
                go.Pie(
                    labels=list(data["response_actions"].keys()),
                    values=list(data["response_actions"].values()),
                    marker_colors=["#2ca02c", "#ff7f0e", "#dc3545", "#7f7f7f"],
                    textinfo="label+percent",
                    hole=0.3,
                )
            ]
        )

        fig_response.update_layout(
            height=400,
            margin={"l": 0, "r": 0, "t": 50, "b": 0},
        )

        st.plotly_chart(fig_response, use_container_width=True)

with tab2:
    st.subheader("Performance Metrics")

    perf = data["performance"]

    # Performance budget table
    ekf_val = f"{perf['ekf_cycle_time_ms']:.2f}"
    hung_val = f"{perf['hungarian_time_ms']:.2f}"
    gnn_val = f"{perf['gnn_inference_ms']:.2f}"
    e2e_val = f"{perf['end_to_end_latency_ms']:.2f}"
    cont_val = f"{perf['track_continuity_rate']:.1%}"
    budgets = {
        "EKF Cycle Time (ms)": {
            "target": "< 1", "actual": ekf_val, "limit": "< 2",
        },
        "Hungarian Assignment (ms)": {
            "target": "< 5", "actual": hung_val, "limit": "< 10",
        },
        "GNN Inference (ms)": {
            "target": "< 20", "actual": gnn_val, "limit": "< 50",
        },
        "End-to-End Latency (ms)": {
            "target": "< 100", "actual": e2e_val, "limit": "< 200",
        },
        "Track Continuity Rate": {
            "target": "> 95%", "actual": cont_val, "limit": "> 90%",
        },
    }



    df_perf = pd.DataFrame(
        [
            {"Metric": k, "Target": v["target"], "Actual": v["actual"], "Hard Limit": v["limit"]}
            for k, v in budgets.items()
        ]
    )

    st.dataframe(df_perf, use_container_width=True, hide_index=True)

    # Performance over time (demo)
    st.markdown("---")
    st.subheader("Latency Over Time")

    np.random.seed(42)
    latency_samples = 80 + np.random.randn(50) * 10
    latency_samples = np.clip(latency_samples, 50, 120)

    fig_latency = go.Figure(
        data=[
            go.Scatter(
                y=latency_samples,
                mode="lines",
                line={"color": "#1f77b4"},
                name="Latency",
            ),
            go.Scatter(
                y=[100] * 50,
                mode="lines",
                line={"color": "green", "dash": "dash"},
                name="Target (100ms)",
            ),
            go.Scatter(
                y=[200] * 50,
                mode="lines",
                line={"color": "red", "dash": "dash"},
                name="Hard Limit (200ms)",
            ),
        ]
    )

    fig_latency.update_layout(
        xaxis={"title": "Sample"},
        yaxis={"title": "Latency (ms)"},
        height=400,
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )

    st.plotly_chart(fig_latency, use_container_width=True)

with tab3:
    st.subheader("Threat Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Threat Level Distribution")

        fig_threat = go.Figure(
            data=[
                go.Bar(
                    x=list(data["threat_dist"].keys()),
                    y=list(data["threat_dist"].values()),
                    marker_color=["#28a745", "#fd7e14", "#dc3545", "#8b0000"],
                    text=[str(v) for v in data["threat_dist"].values()],
                    textposition="auto",
                )
            ]
        )

        fig_threat.update_layout(
            xaxis={"title": "Threat Level"},
            yaxis={"title": "Count"},
            height=400,
            showlegend=False,
            margin={"l": 0, "r": 0, "t": 50, "b": 0},
        )

        st.plotly_chart(fig_threat, use_container_width=True)

    with col2:
        st.subheader("Threat Score Trend")

        # Generate threat score trend
        np.random.seed(42)
        threat_scores = 0.4 + 0.3 * np.sin(np.arange(24) * np.pi / 6) + np.random.randn(24) * 0.1
        threat_scores = np.clip(threat_scores, 0, 1)

        fig_trend = go.Figure(
            data=[
                go.Scatter(
                    x=np.arange(24),
                    y=threat_scores,
                    mode="lines+markers",
                    line={"color": "#dc3545", "width": 2},
                    fill="tozeroy",
                    fillcolor="rgba(220, 53, 69, 0.2)",
                    name="Avg Threat Score",
                )
            ]
        )

        fig_trend.update_layout(
            xaxis={"title": "Hour", "range": [0, 23]},
            yaxis={"title": "Threat Score", "range": [0, 1]},
            height=400,
            margin={"l": 0, "r": 0, "t": 50, "b": 0},
        )

        st.plotly_chart(fig_trend, use_container_width=True)

    # High-threat events log
    st.markdown("---")
    st.subheader("High-Threat Events Log")

    events = [
        {
            "Time": "2024-01-15 14:32:15",
            "Swarm": "swarm_001",
            "Threat": "HIGH",
            "Action": "INTERCEPT",
        },
        {
            "Time": "2024-01-15 12:15:00",
            "Swarm": "swarm_003",
            "Threat": "CRITICAL",
            "Action": "ESCALATE",
        },
        {
            "Time": "2024-01-15 09:45:30",
            "Swarm": "swarm_002",
            "Threat": "HIGH",
            "Action": "JAM",
        },
    ]

    df_events = pd.DataFrame(events)
    st.dataframe(df_events, use_container_width=True, hide_index=True)
