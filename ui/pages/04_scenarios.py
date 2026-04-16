"""Scenario Launcher - Demo scenario management."""

from __future__ import annotations

import streamlit as st

from ui.theme import apply_theme

apply_theme()

st.set_page_config(page_title="Scenarios", layout="wide")

st.title("🎯 Simulation Scenarios")
st.markdown("Launch and manage demo scenarios for testing and demonstration.")

# Scenario definitions
SCENARIOS = {
    "SINGLE_SWARM_TRANSIT": {
        "name": "Single Swarm Transit",
        "description": "A single swarm of 15 drones moving cohesively toward a target area.",
        "difficulty": "Easy",
        "n_drones": 15,
        "n_swarms": 1,
        "threat_level": "Medium",
    },
    "SATURATION_ATTACK": {
        "name": "Saturation Attack",
        "description": "High-speed attack swarm with 25 drones in tight formation.",
        "difficulty": "Hard",
        "n_drones": 25,
        "n_swarms": 1,
        "threat_level": "High",
    },
    "DECOY_AND_STRIKE": {
        "name": "Decoy & Strike",
        "description": "A scattered decoy swarm distracts while a hidden strike force approaches.",
        "difficulty": "Hard",
        "n_drones": 18,
        "n_swarms": 2,
        "threat_level": "High",
    },
    "ENCIRCLEMENT": {
        "name": "Encirclement",
        "description": "Two swarms perform a pincer movement to surround the target.",
        "difficulty": "Medium",
        "n_drones": 24,
        "n_swarms": 2,
        "threat_level": "High",
    },
    "MULTI_SWARM_COORDINATED": {
        "name": "Multi-Swarm Coordinated",
        "description": "Three swarms with different behaviors coordinate their approach.",
        "difficulty": "Very Hard",
        "n_drones": 28,
        "n_swarms": 3,
        "threat_level": "Critical",
    },
    "SCATTER_MANEUVER": {
        "name": "Scatter Maneuver",
        "description": "20 drones perform a saturation dispersal maneuver.",
        "difficulty": "Medium",
        "n_drones": 20,
        "n_swarms": 1,
        "threat_level": "Medium",
    },
}

# Scenario status
if "active_scenario" not in st.session_state:
    st.session_state.active_scenario = None

if "scenario_history" not in st.session_state:
    st.session_state.scenario_history = []

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Available Scenarios")

    for _scenario_key, scenario in SCENARIOS.items():
        # Difficulty color
        diff_colors = {
            "Easy": "green",
            "Medium": "orange",
            "Hard": "red",
            "Very Hard": "darkred",
        }
        diff_color = diff_colors.get(scenario["difficulty"], "gray")

        # Threat color
        threat_colors = {
            "Low": "green",
            "Medium": "orange",
            "High": "red",
            "Critical": "darkred",
        }
        threat_color = threat_colors.get(scenario["threat_level"], "gray")

        # Create card
        with st.container():
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                    background: white;
                ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3 style="margin: 0;">{scenario['name']}</h3>
                        <p style="margin: 5px 0; color: #666;">{scenario['description']}</p>
                        <div style="margin-top: 10px;">
                            <span style="
                                background: {diff_color};
                                color: white;
                                padding: 2px 8px;
                                border-radius: 4px;
                                font-size: 12px;
                            ">Difficulty: {scenario['difficulty']}</span>
                            <span style="
                                background: {threat_color};
                                color: white;
                                padding: 2px 8px;
                                border-radius: 4px;
                                font-size: 12px;
                                margin-left: 8px;
                            ">Threat: {scenario['threat_level']}</span>
                        </div>
                        <div style="margin-top: 8px; font-size: 14px; color: #666;">
                            📍 {scenario['n_drones']} drones | 🌐 {scenario['n_swarms']} swarms
                        </div>
                    </div>
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with col2:
    st.subheader("Launch Scenario")

    selected = st.selectbox(
        "Select Scenario",
        [SCENARIOS[k]["name"] for k in SCENARIOS],
    )

    # Find the key for selected scenario
    selected_key = None
    for key, val in SCENARIOS.items():
        if val["name"] == selected:
            selected_key = key
            break

    if selected_key:
        scenario = SCENARIOS[selected_key]
        st.markdown(
            f"""
            <div style="
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
            ">
            <b>Preview:</b><br>
            Drones: {scenario['n_drones']}<br>
            Swarms: {scenario['n_swarms']}<br>
            Expected Threat: {scenario['threat_level']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🚀 Launch Scenario", use_container_width=True, type="primary"):
            st.session_state.active_scenario = selected_key
            st.session_state.scenario_history.append(selected_key)
            st.success(f"Launched scenario: {scenario['name']}")

    st.markdown("---")
    st.subheader("Current Status")

    if st.session_state.active_scenario:
        active = SCENARIOS[st.session_state.active_scenario]
        st.info(f"Active: {active['name']}")
    else:
        st.warning("No scenario active")

    if st.button("⏹️ Stop Scenario", use_container_width=True):
        st.session_state.active_scenario = None
        st.success("Scenario stopped")

# History
st.markdown("---")
st.subheader("Scenario History")

if st.session_state.scenario_history:
    history_data = []
    for i, key in enumerate(st.session_state.scenario_history):
        scenario = SCENARIOS[key]
        history_data.append(
            {
                "#": i + 1,
                "Scenario": scenario["name"],
                "Drones": scenario["n_drones"],
                "Swarms": scenario["n_swarms"],
                "Threat": scenario["threat_level"],
            }
        )

    import pandas as pd

    df = pd.DataFrame(history_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No scenarios launched yet")
