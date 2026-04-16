# UI_DESIGN.md — Streamlit UI Design System
# AEGIS-AI Platform

---

## 1. Design Principles

- **Light theme throughout** — white/light-gray background, dark text, colored accents
- **Information density > decoration** — operators need data, not chrome
- **Color encodes meaning** — behavior classes have fixed colors used consistently everywhere
- **Real-time first** — every element that can refresh, does refresh automatically
- **Demo-ready** — simulation data always available, hardware optional

---

## 2. Color System

```python
# ui/theme.py  — single source of truth for all colors

COLORS = {
    # Behavior classes (used in ALL charts, maps, tables)
    "TRANSIT":   "#378add",   # blue
    "ATTACK":    "#e24b4a",   # red
    "SCATTER":   "#ef9f27",   # amber
    "ENCIRCLE":  "#9333ea",   # purple
    "DECOY":     "#888780",   # gray
    "UNKNOWN":   "#b4b2a9",   # light gray

    # Alert levels
    "CRITICAL":  "#e24b4a",
    "WARNING":   "#ef9f27",
    "INFO":      "#378add",
    "SAFE":      "#1d9e75",

    # UI surfaces (light theme)
    "BG_PRIMARY":    "#ffffff",
    "BG_SECONDARY":  "#f8f9fa",
    "BG_TERTIARY":   "#f1efe8",
    "BORDER":        "rgba(0,0,0,0.12)",

    # Text
    "TEXT_PRIMARY":   "#1a1a18",
    "TEXT_SECONDARY": "#5f5e5a",
    "TEXT_MUTED":     "#888780",

    # Map / radar
    "MAP_BG":         "#f8f9fb",
    "MAP_GRID":       "rgba(55,138,221,0.08)",
    "RADAR_SWEEP":    "rgba(29,158,117,0.12)",
    "TRACK_TRAIL":    "rgba(55,138,221,0.35)",
    "UNCERTAINTY":    "rgba(55,138,221,0.15)",
    "JAM_ZONE":       "rgba(29,158,117,0.20)",
    "SWARM_HULL":     "rgba(226,75,74,0.10)",

    # Accent
    "ACCENT":         "#1d9e75",
    "ACCENT_LIGHT":   "#e1f5ee",
}

BEHAVIOR_COLORS = {
    BehaviorClass.TRANSIT:  COLORS["TRANSIT"],
    BehaviorClass.ATTACK:   COLORS["ATTACK"],
    BehaviorClass.SCATTER:  COLORS["SCATTER"],
    BehaviorClass.ENCIRCLE: COLORS["ENCIRCLE"],
    BehaviorClass.DECOY:    COLORS["DECOY"],
    BehaviorClass.UNKNOWN:  COLORS["UNKNOWN"],
}

# Plotly layout defaults — apply to every figure
PLOTLY_LAYOUT = {
    "paper_bgcolor": "white",
    "plot_bgcolor":  COLORS["MAP_BG"],
    "font":          {"family": "Inter, system-ui, sans-serif",
                      "color": COLORS["TEXT_PRIMARY"], "size": 12},
    "margin":        {"l": 40, "r": 20, "t": 30, "b": 30},
    "showlegend":    True,
    "legend":        {"bgcolor": "rgba(255,255,255,0.9)",
                      "bordercolor": COLORS["BORDER"], "borderwidth": 0.5},
}

# Streamlit config (put in .streamlit/config.toml)
STREAMLIT_CONFIG = """
[theme]
base = "light"
primaryColor = "#1d9e75"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8f9fa"
textColor = "#1a1a18"
font = "sans serif"
"""
```

---

## 3. Page Structure Template

```python
# Every page follows this structure:

import streamlit as st
from ui.theme import COLORS, PLOTLY_LAYOUT
from ui.components.data_client import get_snapshot

st.set_page_config(
    layout="wide",
    page_title="AEGIS-AI — [Page Name]",
    page_icon="⬡",
)

# Sidebar (consistent across all pages)
with st.sidebar:
    st.image("ui/assets/logo.svg", width=120)
    st.markdown("---")
    st.subheader("System Status")
    # service health indicators here
    st.markdown("---")
    st.subheader("Controls")
    # page-specific controls here
    st.markdown("---")
    refresh_rate = st.slider("Refresh rate (s)", 0.5, 5.0, 0.5, 0.5)

# Main content
snapshot = get_snapshot()
if snapshot is None:
    st.warning("Waiting for data... Is the simulator running?")
    st.info("Run `make dev` to start all services.")
    st.stop()

# ... page content ...

# Auto-refresh
import time
time.sleep(refresh_rate)
st.rerun()
```

---

## 4. Radar Map Component Spec

```python
# ui/components/radar_map.py
# Primary visualization: top-down radar/map view

def render_radar_map(snapshot: UISnapshot) -> go.Figure:
    """
    Renders the main tactical radar display.

    Layers (bottom to top):
    1. Background grid (subtle blue lines)
    2. Sensor range rings (dashed circles)
    3. Jam zones (green filled circles, pulsing opacity)
    4. Swarm convex hulls (light red fill, per-swarm)
    5. EKF uncertainty ellipses (light blue, 1-sigma contour)
    6. Drone tracks / trails (colored by behavior, dashed line)
    7. Drone markers (colored circles, sized by speed)
    8. Predicted trajectories (dashed, 30s lookahead)
    9. Sensor positions (green diamonds)
    10. Labels (track IDs, swarm IDs)
    """

    fig = go.Figure()
    fig.update_layout(**PLOTLY_LAYOUT,
                      xaxis=dict(range=[-2000, 2000], ...),
                      yaxis=dict(range=[-2000, 2000], ...))

    # Layer 4: Swarm convex hulls
    for swarm in snapshot.swarms:
        hull_pts = compute_convex_hull(swarm)
        fig.add_trace(go.Scatter(
            x=hull_pts[:, 0], y=hull_pts[:, 1],
            fill="toself",
            fillcolor=COLORS["SWARM_HULL"],
            line=dict(color=BEHAVIOR_COLORS[swarm.behavior_class], width=1.5, dash="dot"),
            name=f"Swarm {swarm.swarm_id}",
        ))

    # Layer 5: Uncertainty ellipses
    for track in snapshot.tracks:
        if track.status == "confirmed":
            ellipse = compute_uncertainty_ellipse(track.state, track.covariance)
            fig.add_trace(go.Scatter(
                x=ellipse[:, 0], y=ellipse[:, 1],
                fill="toself",
                fillcolor=COLORS["UNCERTAINTY"],
                line=dict(color=COLORS["TRACK_TRAIL"], width=0.5),
                showlegend=False,
            ))

    # Layer 7: Drone markers
    for behavior, group in group_by_behavior(snapshot.tracks):
        fig.add_trace(go.Scatter(
            x=[t.state[0] for t in group],
            y=[t.state[1] for t in group],
            mode="markers",
            marker=dict(
                size=10,
                color=BEHAVIOR_COLORS[behavior],
                symbol="circle",
                line=dict(color="white", width=1),
            ),
            name=behavior.name,
        ))

    return fig
```

---

## 5. Swarm Topology Graph Component Spec

```python
# ui/components/swarm_topology.py
# Real-time graph visualization of swarm internal structure

def render_swarm_graph(swarm: SwarmIntelReport) -> go.Figure:
    """
    Plotly network graph showing:
    - Nodes: drones (size = speed magnitude, color = attention weight)
    - Edges: interaction links (weight = attention coefficient α_ij)
    - Highlighted: leader nodes (top-3 by out-attention)
    - Color scale: green (low influence) → red (high influence / leader)
    """

    # Use Plotly's scatter for nodes + lines for edges
    # Node positions: use spring layout from networkx
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from(swarm.graph_edges)
    pos = nx.spring_layout(G, seed=42, k=0.5)

    # Edge traces
    edge_x, edge_y = [], []
    for u, v in swarm.graph_edges:
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(color=COLORS["BORDER"], width=0.8)))

    # Node traces
    node_x = [pos[n][0] for n in G.nodes]
    node_y = [pos[n][1] for n in G.nodes]
    # Color by attention weight (approximated from degree centrality)
    centrality = nx.degree_centrality(G)
    node_colors = [centrality[n] for n in G.nodes]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=14, color=node_colors,
                    colorscale=[[0,"#1d9e75"],[0.5,"#ef9f27"],[1,"#e24b4a"]],
                    showscale=True,
                    colorbar=dict(title="Influence", thickness=10)),
        text=[n[-4:] for n in G.nodes],  # last 4 chars of drone ID
        textposition="top center",
    ))
```

---

## 6. Behavior Timeline Component

```python
# ui/components/behavior_timeline.py
# Stacked bar / Gantt-style timeline of behavior class per swarm

def render_behavior_timeline(events: list[BehaviorEvent]) -> go.Figure:
    """
    X-axis: time
    Y-axis: swarm IDs
    Color bands: behavior class (fixed color mapping)
    Shows when each swarm transitioned between behaviors
    """
    fig = px.timeline(
        df_events,
        x_start="start", x_end="end",
        y="swarm_id",
        color="behavior",
        color_discrete_map={b.name: BEHAVIOR_COLORS[b] for b in BehaviorClass},
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig
```

---

## 7. Scenario Launcher Page Design

```
┌────────────────────────────────────────────────────────────────────┐
│  Scenario Launcher                                                 │
├─────────────────────────────┬──────────────────────────────────────┤
│  SELECT SCENARIO            │  CONFIGURATION                       │
│  ┌─────────────────────┐    │  Drones: ████████░░  50              │
│  │ ● SCN-01 Transit    │    │  Noise:  ████░░░░░░  0.5m            │
│  │ ○ SCN-02 Saturation │    │  Speed:  ██████░░░░  1.2x            │
│  │ ○ SCN-03 Pincer     │    │  Seed:   42                          │
│  │ ○ SCN-04 Decoy      │    │                                      │
│  │ ○ SCN-05 Adaptive   │    │  [▶ Launch]  [⏸ Pause]  [■ Stop]    │
│  └─────────────────────┘    │                                      │
├─────────────────────────────┴──────────────────────────────────────┤
│  LIVE STATISTICS (updates every 500ms)                             │
│  Drones: 47/50  Tracks: 47  Swarms: 1  Behavior: TRANSIT → ATTACK │
│  Track Continuity: 97.8%    Behavior Accuracy: 91.2%               │
├────────────────────────────────────────────────────────────────────┤
│  SCENARIO DESCRIPTION                                              │
│  50 drones approach in transit formation, then scatter into a     │
│  saturation attack pattern at T+30s. Tests swarm grouper          │
│  stability and behavior transition detection.                      │
└────────────────────────────────────────────────────────────────────┘
```

---

## 8. Threat Panel Component

```python
# ui/components/threat_panel.py
# Right-side threat ranking display

def render_threat_panel(assessment: ThreatAssessment):
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Active Swarms", len(assessment.ranked_threats))
    with col2: st.metric("CRITICAL Alerts",
                         sum(1 for t in assessment.ranked_threats
                             if t.alert_level == AlertLevel.CRITICAL),
                         delta_color="inverse")
    with col3: st.metric("Top Threat Score",
                         f"{assessment.ranked_threats[0].threat_score:.0f}/100"
                         if assessment.ranked_threats else "—")

    st.markdown("---")

    # Ranked threat list
    for i, threat in enumerate(assessment.ranked_threats[:10]):
        color = COLORS[threat.alert_level.value.upper()]
        st.markdown(f"""
        <div style="border-left: 3px solid {color}; padding: 6px 12px;
                    margin: 4px 0; background: {COLORS['BG_SECONDARY']}">
          <strong>#{i+1} {threat.swarm_id}</strong>
          <span style="float:right; color:{color}; font-weight:600">
            {threat.threat_score:.0f}
          </span><br>
          <small style="color:{COLORS['TEXT_MUTED']}">
            {threat.countermeasure.value} recommended
          </small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Alert feed
    st.subheader("Recent Alerts")
    for alert in assessment.alerts[-8:]:
        icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}[alert.level]
        st.markdown(f"{icon} `{alert.swarm_id}` — {alert.message}")
```

---

## 9. Assets Required

```
ui/assets/
├── logo.svg                   ← AEGIS-AI hexagon logo (green, simple)
├── favicon.ico
└── style.css                  ← Custom CSS injected via st.markdown

.streamlit/
└── config.toml                ← Theme configuration (see COLORS section)
```

### Custom CSS (injected on every page)
```css
/* ui/assets/style.css */

/* Remove Streamlit default top padding */
.block-container { padding-top: 1rem !important; }

/* Metric card styling */
[data-testid="stMetric"] {
    background: #f8f9fa;
    border: 0.5px solid rgba(0,0,0,0.10);
    border-radius: 8px;
    padding: 0.75rem 1rem;
}

/* Alert-level colored metric values */
.metric-critical [data-testid="stMetricValue"] { color: #e24b4a !important; }
.metric-warning  [data-testid="stMetricValue"] { color: #ef9f27 !important; }
.metric-safe     [data-testid="stMetricValue"] { color: #1d9e75 !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f8f9fa;
    border-right: 0.5px solid rgba(0,0,0,0.10);
}
```
