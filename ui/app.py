"""AEGIS-AI Streamlit main application."""

import streamlit as st

from ui.theme import apply_theme

# Apply theme
apply_theme()

# Main page configuration
st.title("🛡️ AEGIS-AI Counter-UAS System")
st.markdown(
    """
    **Adaptive Electronic Guard and Intelligence System**

    Real-time multi-agent swarm intelligence for tracking, classifying,
    and coordinating responses against hostile drone swarms.
    """
)

# Sidebar navigation
st.sidebar.image("https://via.placeholder.com/200x80?text=AEGIS-AI", use_container_width=True)
st.sidebar.markdown("---")

# System status
st.sidebar.subheader("System Status")
status_col1, status_col2 = st.sidebar.columns(2)
with status_col1:
    st.metric("Services", "4/4", delta="Online")
with status_col2:
    st.metric("Active Tracks", "0", delta="Waiting for data")

st.sidebar.markdown("---")

# Quick actions
st.sidebar.subheader("Quick Actions")
if st.sidebar.button("🚀 Launch Demo Scenario", use_container_width=True):
    st.session_state["launch_scenario"] = True
    st.sidebar.success("Scenario launched!")

if st.sidebar.button("🔄 Reset System", use_container_width=True):
    st.session_state["reset"] = True
    st.sidebar.info("System reset requested")

# Main dashboard
st.subheader("Live Overview")

# Create placeholder for metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Tracked Targets",
        value="0",
        help="Number of individual drones being tracked",
    )

with col2:
    st.metric(
        label="Active Swarms",
        value="0",
        help="Number of detected swarm formations",
    )

with col3:
    st.metric(
        label="Highest Threat",
        value="LOW",
        delta=None,
        help="Highest threat level among detected swarms",
    )

with col4:
    st.metric(
        label="System Uptime",
        value="0:00",
        help="Time since system start",
    )

# Content area
st.markdown("---")

st.subheader("Getting Started")

st.markdown(
    """
    ### Navigation

    - **📡 Live Ops** - Real-time tactical display with track positions
    - **🕸️ Swarm Graph** - Visualize swarm topology and connections
    - **🧠 Behavior** - Swarm behavior classification dashboard
    - **🎯 Scenarios** - Launch and manage demo scenarios
    - **📊 Analytics** - Historical performance and statistics

    ### System Architecture

    AEGIS-AI uses a three-tier agent architecture:

    1. **Tier 1 (Tracker)** - Individual drone tracking using EKF
    2. **Tier 2 (Swarm Intelligence)** - Behavior classification using GNN
    3. **Tier 3 (Coordinator)** - Response coordination using PPO

    ### Data Flow

    ```
    Sensors → Ingest → Tracker → Swarm Intel → Coordinator → Response
    ```
    """
)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
    AEGIS-AI v1.0.0 | Multi-Agent Counter-UAS Platform
    </div>
    """,
    unsafe_allow_html=True,
)
