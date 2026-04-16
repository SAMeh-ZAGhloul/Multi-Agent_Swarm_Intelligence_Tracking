# ARCHITECTURE.md — AEGIS-AI System Architecture

---

## 1. Architectural Philosophy

AEGIS-AI is built on one insight: **the optimal counter to a swarm is another swarm**.

A hostile drone swarm is a distributed system with emergent collective intelligence. A
centralized, single-process tracker cannot match it — it becomes a bottleneck and a
single point of failure. AEGIS-AI therefore deploys a *society of AI agents* that mirrors
the structure of the threat: many lightweight, cooperative agents that collectively maintain
global situational awareness.

### Three Architectural Invariants
1. **No Single Point of Failure** — every critical function is handled by a pool of agents,
   not one process
2. **Computation Follows the Threat** — agent pools scale up when threat count rises
3. **Algorithms Mirror the Math** — code structure maps 1:1 to the mathematical architecture
   in the concept paper

---

## 2. System Layers

```
╔══════════════════════════════════════════════════════════════════════╗
║                        LAYER 0: SENSORS                              ║
║  RF / SDR Sensors   Acoustic Arrays   Computer Vision (YOLOv8)       ║
║  [simulated in demo mode via core/simulation/]                       ║
╚══════════════════════════╦═══════════════════════════════════════════╝
                           ║  Raw observations (position, signal, image bbox)
                           ▼
╔══════════════════════════════════════════════════════════════════════╗
║                  LAYER 1: INGEST & FUSION SERVICE                    ║
║                                                                      ║
║  • Receives raw sensor observations via ZMQ push sockets             ║
║  • Applies sensor-specific noise models                              ║
║  • Performs Covariance Intersection fusion across modalities         ║
║  • Outputs unified ObservationFrame at fixed 10Hz tick               ║
║  • Publishes to Redis stream: aegis:observations                     ║
╚══════════════════════════╦═══════════════════════════════════════════╝
                           ║  Fused ObservationFrame (10Hz)
                           ▼
╔══════════════════════════════════════════════════════════════════════╗
║              TIER 1: INDIVIDUAL DRONE TRACKING AGENTS                ║
║                                                                      ║
║  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  ║
║  │ Tracker     │  │ Tracker     │  │ Tracker     │  ... (N workers) ║
║  │ Agent #1    │  │ Agent #2    │  │ Agent #3    │                  ║
║  │             │  │             │  │             │                  ║
║  │ EKF state   │  │ EKF state   │  │ EKF state   │                  ║
║  │ for drones  │  │ for drones  │  │ for drones  │                  ║
║  │ partition A │  │ partition B │  │ partition C │                  ║
║  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                  ║
║         │                │                │                          ║
║  • Hungarian data association (LAPJV O(N²))                          ║
║  • CTRA-EKF predict/update cycle @ 10Hz                             ║
║  • Track lifecycle management (tentative→confirmed→lost)             ║
║  • Publishes TrackState to Redis: aegis:tracks                       ║
╚══════════════════════════╦═══════════════════════════════════════════╝
                           ║  Confirmed TrackStates (all drones)
                           ▼
╔══════════════════════════════════════════════════════════════════════╗
║              TIER 2: SWARM INTELLIGENCE AGENTS                       ║
║                                                                      ║
║  ┌──────────────────────┐   ┌──────────────────────┐                ║
║  │  Swarm Agent         │   │  Swarm Agent         │  ... (1 per   ║
║  │  swarm_id = "alpha"  │   │  swarm_id = "beta"   │   detected    ║
║  │                      │   │                      │   swarm)      ║
║  │  • Builds swarm      │   │  • Builds swarm      │               ║
║  │    graph G=(V,E)     │   │    graph G=(V,E)     │               ║
║  │  • Runs GAT-GNN      │   │  • Runs GAT-GNN      │               ║
║  │  • Reynolds inversion│   │  • Reynolds inversion│               ║
║  │  • Behavior classify │   │  • Behavior classify │               ║
║  │  • Trajectory predict│   │  • Trajectory predict│               ║
║  └──────────┬───────────┘   └──────────┬───────────┘               ║
║             │                          │                             ║
║  SwarmGrouper: spatial DBSCAN clustering of tracks → swarm groups    ║
║  Publishes SwarmIntelReport to Redis: aegis:swarm_intel              ║
╚══════════════════════════╦═══════════════════════════════════════════╝
                           ║  SwarmIntelReports (per swarm, 2Hz)
                           ▼
╔══════════════════════════════════════════════════════════════════════╗
║              TIER 3: RESPONSE COORDINATOR                            ║
║                                                                      ║
║  • Aggregates all SwarmIntelReports                                  ║
║  • Computes composite ThreatScore per swarm                          ║
║  • Ranks threat queue                                                ║
║  • PPO policy: allocates countermeasures under constraints           ║
║  • Generates AlertEvents for threshold crossings                     ║
║  • Publishes ThreatAssessment to Redis: aegis:threat                 ║
╚══════════════════════════╦═══════════════════════════════════════════╝
                           ║  ThreatAssessment (1Hz)
                           ▼
╔══════════════════════════════════════════════════════════════════════╗
║                     GATEWAY SERVICE (FastAPI)                        ║
║                                                                      ║
║  • Subscribes to all Redis streams                                   ║
║  • Aggregates state into UISnapshot every 500ms                      ║
║  • Serves WebSocket endpoint ws://localhost:8000/ws/live             ║
║  • Serves REST endpoints for scenario control, history, config       ║
╚══════════════════════════╦═══════════════════════════════════════════╝
                           ║  WebSocket + REST
                           ▼
╔══════════════════════════════════════════════════════════════════════╗
║                   STREAMLIT UI (Port 8501)                           ║
║                                                                      ║
║  Page 1: Live Ops       — radar map + threat panel + alerts          ║
║  Page 2: Swarm Graph    — real-time topology visualization           ║
║  Page 3: Behavior       — classification timeline + Reynolds params  ║
║  Page 4: Scenarios      — demo launcher + controls                   ║
║  Page 5: Analytics      — performance metrics + track statistics     ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 3. Data Flow Diagram

```
Simulator / Hardware Sensors
         │
         │ ObservationMessage (ZMQ PUSH)
         ▼
    [ingest service]
    ├── CI Fusion
    └── ObservationFrame ──► Redis Stream: aegis:observations
                                    │
                    ┌───────────────┘
                    │ (10Hz consumer group)
                    ▼
    [tracker service — worker pool]
    ├── LAPJV Assignment
    ├── EKF predict/update (per drone)
    └── TrackState ──────────────────► Redis Stream: aegis:tracks
                                              │
                           ┌──────────────────┘
                           │ (2Hz consumer group)
                           ▼
    [swarm_intel service]
    ├── DBSCAN grouper ──► SwarmGroup events ──► Redis: aegis:swarm_groups
    ├── Graph builder
    ├── GNN classifier
    ├── Reynolds inverter
    └── SwarmIntelReport ─────────────────────► Redis Stream: aegis:swarm_intel
                                                         │
                                          ┌──────────────┘
                                          │ (1Hz)
                                          ▼
    [coordinator service]
    ├── ThreatScore computation
    ├── PPO allocation
    └── ThreatAssessment ─────────────────────────────► Redis: aegis:threat

    [All streams] ──► [gateway service] ──► WebSocket ──► Streamlit UI
                   │
                   └──► PostgreSQL (track history, events, scenario logs)
```

---

## 4. Tier-1: Tracking Agent — Internal Design

```
┌─────────────────────────────────────────────────────┐
│                   TrackerAgent                      │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │  Observation Queue (asyncio.Queue)          │   │
│  │  Filled by Redis consumer @ 10Hz            │   │
│  └──────────────────────┬──────────────────────┘   │
│                         │                           │
│  ┌──────────────────────▼──────────────────────┐   │
│  │  LAPJV Assignment                           │   │
│  │  Cost matrix C[i,j] = Mahalanobis(z_j, x_i)│   │
│  │  Gate: chi2(dof=3, p=0.99) = 11.34         │   │
│  └──────────────────────┬──────────────────────┘   │
│                         │                           │
│         ┌───────────────┴───────────────┐           │
│         │ Assigned        │ Unassigned  │           │
│         ▼                 ▼             ▼           │
│  ┌─────────────┐   ┌──────────┐  ┌──────────────┐  │
│  │ EKF Update  │   │ EKF      │  │ New Track    │  │
│  │ (matched    │   │ Predict  │  │ Init (M/N    │  │
│  │  tracks)    │   │ only     │  │  confirm)    │  │
│  └─────────────┘   └──────────┘  └──────────────┘  │
│                                                     │
│  Track Lifecycle FSM:                               │
│  TENTATIVE(2) → CONFIRMED → COASTING(5) → DELETED  │
│  (confirmed after 2 hits, deleted after 5 misses)  │
│                                                     │
│  Output: List[TrackState] → Redis aegis:tracks     │
└─────────────────────────────────────────────────────┘
```

### EKF State and Matrices

```python
# State: x = [px, py, pz, vx, vy, vz, omega]  shape: (7,)
# CTRA (Constant Turn Rate and Acceleration) model

# State transition (nonlinear — Jacobian used in EKF):
# px' = px + (vx*sin(ω·T) - vy*(1-cos(ω·T))) / ω
# py' = py + (vx*(1-cos(ω·T)) + vy*sin(ω·T)) / ω
# pz' = pz + vz*T
# vx' = vx*cos(ω·T) - vy*sin(ω·T)
# vy' = vx*sin(ω·T) + vy*cos(ω·T)
# vz' = vz
# ω'  = ω

# Observation model (radar/fusion gives position):
# H = [[1,0,0,0,0,0,0],
#       [0,1,0,0,0,0,0],
#       [0,0,1,0,0,0,0]]   shape: (3,7)

# Process noise Q (tuned for consumer drone dynamics):
# σ_acceleration = 2.0 m/s²
# σ_turn_rate    = 0.1 rad/s

# Measurement noise R (from sensor fusion output):
# Diagonal [σ_x², σ_y², σ_z²] = [1.0, 1.0, 2.0] m²
```

---

## 5. Tier-2: Swarm Intelligence Agent — Internal Design

```
┌─────────────────────────────────────────────────────────┐
│                  SwarmIntelAgent                        │
│                  (one instance per swarm)               │
│                                                         │
│  Input: List[TrackState] for drones in this_swarm_id   │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  1. GRAPH CONSTRUCTION (every 500ms)              │  │
│  │                                                   │  │
│  │  Nodes V: one per drone                           │  │
│  │  Node features: [px,py,pz,vx,vy,vz,rf,size] (8)  │  │
│  │  Edges E: connect if dist(i,j) < r_interact=150m  │  │
│  │  Edge features: [dist, rel_vel, bearing] (3)      │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         │                               │
│  ┌──────────────────────▼────────────────────────────┐  │
│  │  2. GNN FORWARD PASS                              │  │
│  │                                                   │  │
│  │  3× Graph Attention layers (GAT)                  │  │
│  │  h_i^(l+1) = σ(Σ_j α_ij W^(l) h_j^(l))          │  │
│  │  Attention: α_ij = softmax(LeakyReLU(a·[Wh_i‖Wh_j]))│
│  │  Global readout: z = Σ_i MLP(h_i^(L))            │  │
│  │  Classifier head: 6-class softmax                 │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         │                               │
│  ┌──────────────────────▼────────────────────────────┐  │
│  │  3. REYNOLDS PARAMETER INVERSION                  │  │
│  │                                                   │  │
│  │  Observed: v_i(t) for all i                       │  │
│  │  Model: v_i = w_s·f_sep + w_a·f_align + w_c·f_coh│  │
│  │  Solve: MLE for [w_s, w_a, w_c] via scipy.minimize│  │
│  │  → Maps to BehaviorClass via lookup table         │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         │                               │
│  ┌──────────────────────▼────────────────────────────┐  │
│  │  4. TRAJECTORY PREDICTION                         │  │
│  │                                                   │  │
│  │  Propagate centroid using estimated Reynolds      │  │
│  │  weights for 30 seconds at dt=1s                  │  │
│  │  Output: List[centroid_position] (30 points)      │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         │                               │
│  Output → SwarmIntelReport → Redis aegis:swarm_intel   │
└─────────────────────────────────────────────────────────┘
```

### Swarm Feature Vector φ(S)
```python
phi = {
    "centroid_pos":          np.array([px, py, pz]),    # (3,)
    "centroid_vel":          np.array([vx, vy, vz]),    # (3,)
    "spread_radius":         float,   # σ = sqrt(Σ|p_i - p_c|² / N)
    "velocity_coherence":    float,   # |Σv_i| / Σ|v_i|  ∈ [0,1]
    "formation_entropy":     float,   # Shannon entropy of bearing distribution
    "convex_hull_area":      float,   # m²
    "leader_count":          int,     # nodes with high out-attention weight
    "approach_rate":         float,   # d(dist_to_asset)/dt  m/s (negative = approaching)
    "reynolds_weights":      np.array([w_sep, w_align, w_coh]),
    "behavior_class":        BehaviorClass,
    "behavior_confidence":   float,
    "trajectory_30s":        np.ndarray,  # shape (30, 3)
}
```

---

## 6. Tier-3: Response Coordinator — Internal Design

```
┌─────────────────────────────────────────────────────────┐
│              ResponseCoordinator                        │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  THREAT SCORING (per swarm, 1Hz)                  │  │
│  │                                                   │  │
│  │  S(k) = w1·P(attack|cls_k)                       │  │
│  │       + w2·(1 - d_k/d_max)                       │  │
│  │       + w3·N_k/N_max                             │  │
│  │       + w4·P(reach_target|traj_k)                │  │
│  │                                                   │  │
│  │  Weights: w=[0.35, 0.30, 0.20, 0.15] (tunable)   │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         │                               │
│  ┌──────────────────────▼────────────────────────────┐  │
│  │  PPO POLICY (trained in simulation)               │  │
│  │                                                   │  │
│  │  State:  threat_vector (20 swarms × 5 features)   │  │
│  │  Action: countermeasure assignment (discrete)     │  │
│  │  Reward: Σ neutralized_k                          │  │
│  │        - λ·cost_k                                │  │
│  │        - μ·collateral_k                          │  │
│  │  γ=0.95, clip_ratio=0.2, entropy_coef=0.01       │  │
│  └──────────────────────┬────────────────────────────┘  │
│                         │                               │
│  Output: ThreatAssessment {                             │
│    ranked_threats: List[ThreatRecord],                  │
│    countermeasure_assignments: Dict[swarm_id, CM],      │
│    alerts: List[AlertEvent],                            │
│    timestamp: float                                     │
│  }                                                      │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Message Schema (Redis Streams)

```python
# aegis:observations  (10Hz per sensor cluster)
ObservationMessage = {
    "obs_id":    str,           # UUID
    "sensor":    str,           # "rf" | "acoustic" | "vision" | "fused"
    "timestamp": float,         # Unix epoch
    "detections": [
        {"det_id": str, "px": float, "py": float, "pz": float,
         "sigma_x": float, "sigma_y": float, "sigma_z": float,
         "rf_dbm": float, "confidence": float}
    ]
}

# aegis:tracks  (10Hz, published by tracker pool)
TrackState = {
    "track_id":    str,
    "swarm_id":    str | None,    # None until grouped
    "timestamp":   float,
    "state":       [7 floats],    # [px,py,pz,vx,vy,vz,omega]
    "covariance":  [49 floats],   # P matrix flattened (7×7)
    "status":      str,           # "tentative"|"confirmed"|"coasting"|"deleted"
    "hits":        int,
    "misses":      int,
}

# aegis:swarm_intel  (2Hz, published by swarm intel agents)
SwarmIntelReport = {
    "swarm_id":             str,
    "timestamp":            float,
    "drone_count":          int,
    "behavior_class":       int,        # BehaviorClass enum value
    "behavior_confidence":  float,
    "reynolds_weights":     [3 floats], # [w_sep, w_align, w_coh]
    "threat_score":         float,      # raw, before coordinator ranking
    "centroid":             [3 floats],
    "spread_radius":        float,
    "velocity_coherence":   float,
    "approach_rate":        float,
    "trajectory_30s":       [[3 floats] × 30],
    "graph_edges":          [[str, str]],  # for topology visualization
}

# aegis:threat  (1Hz, published by coordinator)
ThreatAssessment = {
    "timestamp": float,
    "ranked_threats": [
        {"swarm_id": str, "threat_score": float, "rank": int,
         "countermeasure": str, "alert_level": str}
    ],
    "alerts": [
        {"level": str, "swarm_id": str, "message": str, "timestamp": float}
    ]
}
```

---

## 8. SwarmGrouper Algorithm (DBSCAN-based)

Drones are grouped into swarms using a modified DBSCAN that incorporates both spatial
proximity and velocity coherence:

```python
# Distance metric for swarm grouping:
# d(i,j) = alpha * ||p_i - p_j|| + beta * ||v_i - v_j||
# alpha = 1.0, beta = 5.0 (velocity weighted more heavily)
# eps = 200m (spatial), min_samples = 3

# Re-run every 2 seconds
# Swarm IDs are persistent: match new clusters to existing swarms
#   by centroid distance (Hungarian matching on centroids)
# Unmatched new clusters → new swarm_id
# Unmatched existing swarms → mark as SPLIT or MERGED event
```

---

## 9. GNN Model Architecture

```
Input Graph: G = (V, E)
  Node features: x_i ∈ R^8
  Edge features: e_ij ∈ R^3

Layer 1: GATConv(in=8,  out=32, heads=4, concat=True)  → h ∈ R^128
         + BatchNorm + ReLU + Dropout(0.1)

Layer 2: GATConv(in=128, out=64, heads=4, concat=True)  → h ∈ R^256
         + BatchNorm + ReLU + Dropout(0.1)

Layer 3: GATConv(in=256, out=64, heads=1, concat=False) → h ∈ R^64
         + BatchNorm + ReLU

Global Readout: global_mean_pool(h) + global_max_pool(h) → z ∈ R^128

Classifier MLP:
  Linear(128, 64) + ReLU
  Linear(64, 32)  + ReLU
  Linear(32, 6)   → logits
  Softmax → P(BehaviorClass)

Training:
  Dataset: 50,000 synthetic swarm snapshots (Reynolds simulator)
  Optimizer: AdamW lr=3e-4, weight_decay=1e-4
  Scheduler: CosineAnnealingLR
  Loss: CrossEntropyLoss + 0.1 * GraphRegularizationLoss
  Augmentation: random node dropout (p=0.1), edge perturbation
```

---

## 10. UI Architecture (Streamlit)

```
ui/app.py
  ├── st.set_page_config(layout="wide", page_title="AEGIS-AI")
  ├── Loads theme from ui/theme.py
  ├── Initializes WebSocket client to gateway
  └── Stores live state in st.session_state.snapshot

ui/pages/01_live_ops.py
  ├── Left column (70%): Plotly radar map
  │     • Scatter traces: one per drone, colored by behavior
  │     • Ellipse traces: EKF uncertainty (Σ contour)
  │     • Polygon traces: swarm convex hull boundaries
  │     • Line traces: predicted trajectories (dashed)
  │     • Marker traces: sensor positions
  ├── Right column (30%): Threat panel
  │     • st.metric cards: active drones, swarms, alerts
  │     • Ranked threat list with color-coded scores
  │     • Alert feed (last 10 events)
  └── Auto-refresh: time.sleep(0.5) + st.rerun()

ui/pages/02_swarm_graph.py
  ├── PyVis network OR Plotly graph (configurable)
  ├── Nodes: drones, sized by velocity, colored by behavior
  ├── Edges: interaction links, weighted by attention α_ij
  ├── Highlighted: leader nodes (top-k by attention weight)
  └── Side panel: selected swarm statistics

ui/pages/03_behavior.py
  ├── Timeline chart: behavior class over time (stacked bar)
  ├── Reynolds weights radar chart (w_sep, w_align, w_coh)
  ├── Confidence gauge per swarm
  └── Manual injection sliders (for demo: override behavior)

ui/pages/04_scenarios.py
  ├── Scenario selector (5 named scenarios)
  ├── Parameter controls (drone count, noise level, speed)
  ├── Launch / Pause / Reset buttons
  ├── Live statistics during run
  └── Post-run performance report

ui/pages/05_analytics.py
  ├── Track continuity rate over time
  ├── Classification accuracy (vs ground truth from simulator)
  ├── Latency histogram (sensor→threat score)
  └── Per-scenario comparison table
```

---

## 11. Dependency Graph (No Circular Dependencies)

```
core/constants.py
    ↑
core/models.py (Pydantic dataclasses)
    ↑                ↑
core/tracking/    core/swarm/       core/simulation/
    ↑                ↑
services/tracker  services/swarm_intel
    ↑                ↑
services/coordinator
    ↑
services/gateway
    ↑
ui/
```

Rule: arrows point upward only. UI never imports from services directly —
only through WebSocket/REST from gateway.
