"""Swarm Graph - Visualize swarm topology."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ui.theme import apply_theme

apply_theme()

st.set_page_config(page_title="Swarm Graph", layout="wide")

st.title("🕸️ Swarm Topology")
st.markdown("Visualize swarm structure and inter-drone connections.")

# Sidebar
st.sidebar.subheader("Graph Options")
show_edges = st.sidebar.checkbox("Show Connections", value=True)
show_labels = st.sidebar.checkbox("Show Labels", value=True)
edge_threshold = st.sidebar.slider("Connection Distance (m)", 10, 100, 50)

# Demo swarm data
def generate_demo_swarm(n_drones: int = 20) -> dict:
    """Generate demo swarm data."""
    # Create a swarm with some structure
    np.random.seed(42)

    # Cluster centers
    centers = [
        np.array([0, 0, 25]),
        np.array([30, 20, 25]),
        np.array([-20, 30, 25]),
    ]

    positions = []
    velocities = []

    for i in range(n_drones):
        center = centers[i % len(centers)]
        offset = np.random.randn(3) * 10
        pos = center + offset
        positions.append(pos)

        # Velocity toward center with noise
        vel = (center - pos) * 0.1 + np.random.randn(3) * 0.5
        velocities.append(vel)

    return {
        "positions": np.array(positions),
        "velocities": np.array(velocities),
        "n_drones": n_drones,
    }


# Generate data
swarm_data = generate_demo_swarm(20)
positions = swarm_data["positions"]
velocities = swarm_data["velocities"]

# Build edges based on distance threshold
edges = []
for i in range(len(positions)):
    for j in range(i + 1, len(positions)):
        dist = np.linalg.norm(positions[i] - positions[j])
        if dist < edge_threshold:
            edges.append((i, j))

# Create visualization
def create_swarm_graph(
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    show_edges: bool = True,
    show_labels: bool = True,
) -> go.Figure:
    """Create swarm topology graph visualization."""
    fig = go.Figure()

    # Add edges
    if show_edges and edges:
        edge_x = []
        edge_y = []
        edge_z = []

        for i, j in edges:
            edge_x.extend([positions[i][0], positions[j][0], None])
            edge_y.extend([positions[i][1], positions[j][1], None])
            edge_z.extend([positions[i][2], positions[j][2], None])

        fig.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line={"color": "#cccccc", "width": 2},
                name="Connections",
                hoverinfo="skip",
            )
        )

    # Add nodes
    node_x = positions[:, 0]
    node_y = positions[:, 1]
    node_z = positions[:, 2]

    # Color by cluster (based on z-height for demo)
    colors = np.where(node_z > 30, "red", np.where(node_z > 25, "orange", "blue"))

    fig.add_trace(
        go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers+text" if show_labels else "markers",
            marker={
                "size": 8,
                "color": colors,
                "line": {"color": "white", "width": 1},
            },
            text=[f"D{i}" for i in range(len(positions))],
            textposition="top center",
            textfont={"size": 10, "color": "black"},
            name="Drones",
            hovertemplate=(
                "<b>Drone %{text}</b><br>"
                + "Position: (%{x:.1f}, %{y:.1f}, %{z:.1f})<br>"
                + "<extra></extra>"
            ),
        )
    )

    # Layout
    fig.update_layout(
        scene={
            "xaxis": {"title": "X (m)"},
            "yaxis": {"title": "Y (m)"},
            "zaxis": {"title": "Z (m)"},
            "aspectmode": "cube",
        },
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        height=600,
        showlegend=True,
    )

    return fig


# Main display
col1, col2 = st.columns([2, 1])

with col1:
    fig = create_swarm_graph(
        positions,
        edges,
        show_edges=show_edges,
        show_labels=show_labels,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Graph Statistics")

    n_nodes = len(positions)
    n_edges = len(edges)
    avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0

    # Graph density
    max_edges = n_nodes * (n_nodes - 1) / 2
    density = n_edges / max_edges if max_edges > 0 else 0

    st.metric("Nodes (Drones)", n_nodes)
    st.metric("Edges (Connections)", n_edges)
    st.metric("Average Degree", f"{avg_degree:.2f}")
    st.metric("Graph Density", f"{density:.3f}")

    st.markdown("---")
    st.subheader("Swarm Metrics")

    # Compute swarm metrics
    centroid = np.mean(positions, axis=0)
    spread = np.mean(np.linalg.norm(positions - centroid, axis=1))
    mean_speed = np.mean(np.linalg.norm(velocities, axis=1))

    st.metric("Centroid (m)", f"({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")
    st.metric("Spread Radius (m)", f"{spread:.1f}")
    st.metric("Mean Speed (m/s)", f"{mean_speed:.2f}")

    # Cohesion metric
    norm_vels = velocities / (np.linalg.norm(velocities, axis=1, keepdims=True) + 1e-6)
    mean_dir = np.mean(norm_vels, axis=0)
    coherence = np.linalg.norm(mean_dir)
    st.metric("Velocity Coherence", f"{coherence:.3f}")

# Adjacency info
st.markdown("---")
st.subheader("Adjacency Information")

adj_data = []
for i in range(len(positions)):
    neighbors = [j for (a, j) in edges if a == i] + [a for (a, b) in edges if b == i]
    adj_data.append(
        {
            "Drone": f"D{i}",
            "Degree": len(neighbors),
            "Neighbors": ", ".join([f"D{j}" for j in neighbors[:5]]),
        }
    )

import pandas as pd

df = pd.DataFrame(adj_data)
st.dataframe(df, use_container_width=True, hide_index=True)
