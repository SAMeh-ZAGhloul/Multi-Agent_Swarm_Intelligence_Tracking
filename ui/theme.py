"""AEGIS-AI UI theme configuration."""

from __future__ import annotations

# Light theme color palette
AEGIS_THEME = {
    # Primary colors
    "primaryColor": "#1f77b4",  # Blue
    "backgroundColor": "#ffffff",  # White
    "secondaryBackgroundColor": "#f0f2f6",  # Light gray

    # Semantic colors
    "threat_high": "#dc3545",  # Red
    "threat_medium": "#fd7e14",  # Orange
    "threat_low": "#28a745",  # Green

    # Chart colors
    "chart_colors": [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
    ],

    # Font
    "font": "sans-serif",
    "headingFont": "sans-serif",
}

# Plotly theme configuration
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "responsive": True,
}

# Map/Radar configuration
MAP_BOUNDS = {
    "x_min": -150,
    "x_max": 150,
    "y_min": -150,
    "y_max": 150,
    "z_min": 0,
    "z_max": 100,
}

# Refresh rates (milliseconds)
REFRESH_RATES = {
    "live_ops": 500,
    "swarm_graph": 1000,
    "behavior": 1000,
    "analytics": 5000,
}


def apply_theme() -> None:
    """Apply AEGIS theme to Streamlit page."""
    import streamlit as st

    st.set_page_config(
        page_title="AEGIS-AI Counter-UAS",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/aegis-ai/aegis-ai",
            "Report a bug": "https://github.com/aegis-ai/aegis-ai/issues",
            "About": "AEGIS-AI: Adaptive Electronic Guard and Intelligence System",
        },
    )

    # Custom CSS for additional styling
    st.markdown(
        """
        <style>
        .main {
            background-color: #ffffff;
        }
        .stAlert {
            border-radius: 0.5rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 1rem;
            padding: 1.5rem;
            color: white;
        }
        .threat-high {
            background-color: #dc3545;
            color: white;
        }
        .threat-medium {
            background-color: #fd7e14;
            color: white;
        }
        .threat-low {
            background-color: #28a745;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
