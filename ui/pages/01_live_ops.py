"""Live Operations - Real-time tactical display."""

from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ui.theme import MAP_BOUNDS, apply_theme

apply_theme()

st.set_page_config(page_title="Live Ops", layout="wide")

st.title("📡 Live Operations")
st.markdown("Real-time tactical display of tracked targets and swarms.")

# Sidebar controls
st.sidebar.subheader("Display Options")
show_tracks = st.sidebar.checkbox("Show Tracks", value=True)
show_swarms = st.sidebar.checkbox("Show Swarms", value=True)
show_velocities = st.sidebar.checkbox("Show Velocity Vectors", value=False)
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)

refresh_rate = st.sidebar.slider("Refresh Rate (Hz)", 1, 10, 2)

# Simulated data (in production, this comes from WebSocket)
if "sim_time" not in st.session_state:
    st.session_state.sim_time = 0.0

if "tracks" not in st.session_state:
    st.session_state.tracks = []

if "swarms" not in st.session_state:
    st.session_state.swarms = []

# Generate demo data
def generate_demo_tracks(n_tracks: int = 10) -> list[dict]:
    """Generate demo track data."""
    tracks = []
    for i in range(n_tracks):
        angle = (i / n_tracks) * 2 * np.pi
        radius = 30 + 10 * np.sin(angle * 3)
        track = {
            "track_id": f"track_{i}",
            "position": [
                radius * np.cos(angle),
                radius * np.sin(angle),
                25 + 5 * np.sin(time.time() + i),
            ],
            "velocity": [
                -np.sin(angle) * 5,
                np.cos(angle) * 5,
                0,
            ],
            "speed": 5.0,
            "confirmed": True,
            "age": i * 10,
        }
        tracks.append(track)
    return tracks


def generate_demo_swarms(tracks: list[dict]) -> list[dict]:
    """Generate demo swarm data from tracks."""
    if not tracks:
        return []

    # Group tracks into swarms (simple clustering)
    swarms = [
        {
            "swarm_id": "swarm_1",
            "n_drones": len(tracks),
            "centroid": np.mean([t["position"] for t in tracks], axis=0).tolist(),
            "behavior": "TRANSIT",
            "threat_score": 0.5,
            "track_ids": [t["track_id"] for t in tracks],
        }
    ]
    return swarms


# Update demo data
if auto_refresh:
    st.session_state.sim_time += 0.1
    st.session_state.tracks = generate_demo_tracks(10)
    st.session_state.swarms = generate_demo_swarms(st.session_state.tracks)

# Create radar/map plot
def create_radar_map(
    tracks: list[dict],
    swarms: list[dict],
    show_tracks: bool = True,
    show_swarms: bool = True,
    show_velocities: bool = False,
) -> go.Figure:
    """Create the main radar/map visualization."""
    fig = go.Figure()

    # Add background grid
    fig.add_trace(
        go.Scatter(
            x=[-MAP_BOUNDS["x_max"], MAP_BOUNDS["x_max"]],
            y=[0, 0],
            mode="lines",
            line={"color": "#eee", "width": 1},
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[-MAP_BOUNDS["y_max"], MAP_BOUNDS["y_max"]],
            mode="lines",
            line={"color": "#eee", "width": 1},
            showlegend=False,
        )
    )

    # Add range rings
    for radius in [50, 100, 150]:
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(
            go.Scatter(
                x=radius * np.cos(theta),
                y=radius * np.sin(theta),
                mode="lines",
                line={"color": "#f0f0f0", "width": 1, "dash": "dash"},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Plot tracks
    if show_tracks and tracks:
        track_x = [t["position"][0] for t in tracks]
        track_y = [t["position"][1] for t in tracks]
        track_z = [t["position"][2] for t in tracks]

        fig.add_trace(
            go.Scatter3d(
                x=track_x,
                y=track_y,
                z=track_z,
                mode="markers+text",
                marker={
                    "size": 8,
                    "color": [t.get("threat_score", 0.5) for t in tracks],
                    "colorscale": "RdYlGn_r",
                    "colorbar": {"title": "Threat"},
                    "line": {"color": "white", "width": 1},
                },
                text=[t["track_id"] for t in tracks],
                textposition="top center",
                name="Tracks",
            )
        )

        # Add velocity vectors
        if show_velocities:
            for track in tracks:
                pos = track["position"]
                vel = track["velocity"]
                fig.add_trace(
                    go.Cone(
                        x=[pos[0]],
                        y=[pos[1]],
                        z=[pos[2]],
                        u=[vel[0]],
                        v=[vel[1]],
                        w=[vel[2]],
                        colorscale="Blues",
                        showscale=False,
                        sizemode="absolute",
                        sizeref=0.5,
                        name="Velocity",
                        hoverinfo="skip",
                    )
                )

    # Plot swarm centroids
    if show_swarms and swarms:
        swarm_x = [s["centroid"][0] for s in swarms]
        swarm_y = [s["centroid"][1] for s in swarms]
        swarm_z = [s["centroid"][2] for s in swarms]

        threat_colors = []
        for s in swarms:
            if s["threat_score"] > 0.7:
                threat_colors.append("red")
            elif s["threat_score"] > 0.4:
                threat_colors.append("orange")
            else:
                threat_colors.append("green")

        fig.add_trace(
            go.Scatter3d(
                x=swarm_x,
                y=swarm_y,
                z=swarm_z,
                mode="markers+text",
                marker={
                    "size": 15,
                    "color": threat_colors,
                    "symbol": "diamond",
                    "line": {"color": "darkred", "width": 2},
                },
                text=[f"⭐ {s['swarm_id']}" for s in swarms],
                textposition="top center",
                name="Swarm Centroids",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    + "Drones: %{customdata[0]}<br>"
                    + "Behavior: %{customdata[1]}<br>"
                    + "Threat: %{customdata[2]:.2f}<extra></extra>"
                ),
                customdata=[
                    [s["n_drones"], s["behavior"], s["threat_score"]] for s in swarms
                ],
            )
        )

    # Layout
    fig.update_layout(
        scene={
            "xaxis": {
                "title": "X (m)",
                "range": [MAP_BOUNDS["x_min"], MAP_BOUNDS["x_max"]],
            },
            "yaxis": {
                "title": "Y (m)",
                "range": [MAP_BOUNDS["y_min"], MAP_BOUNDS["y_max"]],
            },
            "zaxis": {
                "title": "Z (m)",
                "range": [MAP_BOUNDS["z_min"], MAP_BOUNDS["z_max"]],
            },
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        height=600,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


# Main display
col1, col2 = st.columns([3, 1])

with col1:
    fig = create_radar_map(
        st.session_state.tracks,
        st.session_state.swarms,
        show_tracks=show_tracks,
        show_swarms=show_swarms,
        show_velocities=show_velocities,
    )
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})

with col2:
    st.subheader("Track Summary")

    if st.session_state.tracks:
        for track in st.session_state.tracks[:5]:
            threat = track.get("threat_score", 0.5)
            if threat > 0.7:
                threat_emoji = "🔴"
            elif threat > 0.4:
                threat_emoji = "🟡"
            else:
                threat_emoji = "🟢"

            st.markdown(
                f"""
                <div style="padding: 8px; margin: 4px 0; border-radius: 4px; background: #f8f9fa;">
                <b>{threat_emoji} {track["track_id"]}</b><br>
                Speed: {track["speed"]:.1f} m/s | Age: {track["age"]}s
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Swarm Summary")
    if st.session_state.swarms:
        for swarm in st.session_state.swarms:
            threat = swarm["threat_score"]
            if threat > 0.7:
                threat_class = "threat-high"
            elif threat > 0.4:
                threat_class = "threat-medium"
            else:
                threat_class = "threat-low"

            st.markdown(
                f"""
                <div style="padding: 8px; margin: 4px 0;
                border-radius: 4px;" class="{threat_class}">
                <b>⚠️ {swarm["swarm_id"]}</b><br>
                Drones: {swarm["n_drones"]} | {swarm["behavior"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

# Data table
st.markdown("---")
st.subheader("Track Data Table")

if st.session_state.tracks:
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "Track ID": t["track_id"],
                "X": f"{t['position'][0]:.1f}",
                "Y": f"{t['position'][1]:.1f}",
                "Z": f"{t['position'][2]:.1f}",
                "Speed (m/s)": f"{t['speed']:.1f}",
                "Age (s)": t["age"],
            }
            for t in st.session_state.tracks
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No tracks currently being tracked")
