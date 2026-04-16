"""Behavior Classification Dashboard."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ui.theme import apply_theme

apply_theme()

st.set_page_config(page_title="Behavior Analysis", layout="wide")

st.title("🧠 Behavior Classification")
st.markdown(
    """
    Swarm behavior analysis using Reynolds flocking parameter inversion
    and graph neural network classification.
    """
)

# Behavior classes
BEHAVIOR_CLASSES = {
    "TRANSIT": {
        "description": "Cohesive movement toward target",
        "weights": [0.3, 0.8, 0.6],
        "threat": 0.5,
        "color": "blue",
    },
    "ATTACK": {
        "description": "Converging, high speed, low spread",
        "weights": [0.8, 0.2, 0.1],
        "threat": 0.95,
        "color": "red",
    },
    "SCATTER": {
        "description": "Dispersing (saturation maneuver)",
        "weights": [0.9, 0.1, 0.1],
        "threat": 0.8,
        "color": "orange",
    },
    "ENCIRCLE": {
        "description": "Pincer / encirclement formation",
        "weights": [0.5, 0.7, 0.9],
        "threat": 0.85,
        "color": "purple",
    },
    "DECOY": {
        "description": "Disordered, high separation",
        "weights": [0.9, 0.1, 0.3],
        "threat": 0.4,
        "color": "gray",
    },
    "UNKNOWN": {
        "description": "Insufficient data for classification",
        "weights": [0.33, 0.33, 0.33],
        "threat": 0.3,
        "color": "lightgray",
    },
}

# Sidebar
st.sidebar.subheader("Classification Settings")
selected_behavior = st.sidebar.selectbox(
    "Simulated Behavior",
    list(BEHAVIOR_CLASSES.keys()),
    index=0,
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.0,
    1.0,
    0.6,
)

# Demo classification data
def generate_classification_data(behavior: str) -> dict:
    """Generate demo classification data."""
    base_weights = BEHAVIOR_CLASSES[behavior]["weights"]

    # Add some noise
    np.random.seed(42)
    weights = np.array(base_weights) + np.random.randn(3) * 0.05
    weights = np.clip(weights, 0, 1)
    weights = weights / weights.sum()  # Normalize

    # Compute confidence based on how distinct the behavior is
    confidence_map = {
        "ATTACK": 0.92,
        "SCATTER": 0.85,
        "ENCIRCLE": 0.78,
        "TRANSIT": 0.88,
        "DECOY": 0.65,
        "UNKNOWN": 0.45,
    }

    return {
        "predicted_behavior": behavior,
        "confidence": confidence_map.get(behavior, 0.5),
        "weights": weights,
        "kinematic_features": {
            "mean_speed": np.random.uniform(5, 15),
            "velocity_coherence": np.random.uniform(0.3, 0.9),
            "spread_radius": np.random.uniform(10, 40),
        },
    }


classification = generate_classification_data(selected_behavior)

# Main display
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Classification Result")

    behavior_info = BEHAVIOR_CLASSES[classification["predicted_behavior"]]

    # Display with color coding
    color_map = {
        "blue": "#1f77b4",
        "red": "#dc3545",
        "orange": "#fd7e14",
        "purple": "#9467bd",
        "gray": "#7f7f7f",
        "lightgray": "#d3d3d3",
    }

    bg_color = color_map.get(behavior_info["color"], "#1f77b4")

    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        ">
        <h2 style="margin: 0;">{classification["predicted_behavior"]}</h2>
        <p style="margin: 10px 0 0 0; opacity: 0.9;">{behavior_info["description"]}</p>
        <p style="margin: 10px 0 0 0; font-size: 1.2em;">
            Confidence: {classification["confidence"]:.1%}
        </p>
        <p style="margin: 10px 0 0 0;">
            Threat Score: {behavior_info["threat"]:.2f}
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Threat level indicator
    threat = behavior_info["threat"]
    if threat > 0.7:
        threat_level = "🔴 HIGH"
    elif threat > 0.4:
        threat_level = "🟡 MEDIUM"
    else:
        threat_level = "🟢 LOW"

    st.metric("Threat Level", threat_level)

with col2:
    st.subheader("Reynolds Weights")

    # Bar chart of weights
    weight_fig = go.Figure(
        data=[
            go.Bar(
                x=["Separation", "Alignment", "Cohesion"],
                y=classification["weights"],
                marker_color=["#ff7f0e", "#2ca02c", "#1f77b4"],
                text=[f"{w:.2f}" for w in classification["weights"]],
                textposition="auto",
            )
        ]
    )

    weight_fig.update_layout(
        yaxis={"range": [0, 1], "title": "Weight"},
        height=300,
        showlegend=False,
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )

    st.plotly_chart(weight_fig, use_container_width=True)

with col3:
    st.subheader("Kinematic Features")

    kinematic = classification["kinematic_features"]

    st.metric("Mean Speed", f"{kinematic['mean_speed']:.1f} m/s")
    st.metric("Velocity Coherence", f"{kinematic['velocity_coherence']:.2f}")
    st.metric("Spread Radius", f"{kinematic['spread_radius']:.1f} m")

    # Radar chart of features
    categories = ["Speed", "Coherence", "Tightness"]
    values = [
        kinematic["mean_speed"] / 15,  # Normalize
        kinematic["velocity_coherence"],
        1 - kinematic["spread_radius"] / 50,  # Invert and normalize
    ]

    radar_fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                line_color="#1f77b4",
            )
        ]
    )

    radar_fig.update_layout(
        polar={"radialaxis": {"visible": True, "range": [0, 1]}},
        height=300,
        showlegend=False,
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )

    st.plotly_chart(radar_fig, use_container_width=True)

# Behavior comparison table
st.markdown("---")
st.subheader("Behavior Reference")

ref_data = []
for name, info in BEHAVIOR_CLASSES.items():
    ref_data.append(
        {
            "Behavior": name,
            "Description": info["description"],
            "Separation": f"{info['weights'][0]:.2f}",
            "Alignment": f"{info['weights'][1]:.2f}",
            "Cohesion": f"{info['weights'][2]:.2f}",
            "Threat Score": f"{info['threat']:.2f}",
        }
    )

import pandas as pd

df = pd.DataFrame(ref_data)
st.dataframe(df, use_container_width=True, hide_index=True)

# GNN Model info
st.markdown("---")
st.subheader("GNN Classifier Info")

st.markdown(
    """
    The swarm behavior classifier uses a Graph Neural Network with:

    - **Input**: Node features (position, velocity, speed) + Edge connectivity
    - **Architecture**: 3-layer Graph Convolutional Network (GCN)
    - **Output**: 6-class behavior classification with confidence scores

    The model is trained on simulated swarm data with various Reynolds flocking
    parameters to recognize different behavioral patterns.
    """
)
