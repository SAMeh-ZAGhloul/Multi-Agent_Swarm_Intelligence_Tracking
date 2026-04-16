# AEGIS-AI — Claude Code Instructions

## Project Identity
**AEGIS-AI** (Adaptive Electronic Guard and Intelligence System) is a real-time counter-UAS
platform that uses multi-agent swarm intelligence to track, classify, and coordinate responses
against hostile drone swarms. It fights swarm intelligence *with* swarm intelligence.

## Quick Start for Claude Code
```bash
# Bootstrap the full environment
make install        # installs all Python deps + pre-commit hooks
make sim            # runs the swarm simulator (generates demo data)
make dev            # starts all backend services + Streamlit UI
make test           # runs full test suite
make lint           # ruff + mypy
```

---

## Repo Layout
```
aegis-ai/
├── CLAUDE.md               ← you are here
├── BRD.md                  ← business & functional requirements
├── ARCHITECTURE.md         ← detailed system architecture
├── INFRASTRUCTURE.md       ← deployment topology & sizing
├── API.md                  ← internal service API contracts
├── TESTING.md              ← test strategy & coverage targets
├── Makefile
├── pyproject.toml
├── docker-compose.yml
│
├── core/                   ← pure-Python algorithmic core (no I/O)
│   ├── tracking/
│   │   ├── ekf.py          ← Extended Kalman Filter (EKF + CTRA model)
│   │   ├── hungarian.py    ← LAPJV data association
│   │   ├── track.py        ← Track object (state, covariance, lifecycle)
│   │   └── fusion.py       ← Covariance Intersection multi-sensor fusion
│   ├── swarm/
│   │   ├── reynolds.py     ← Reynolds flocking model + parameter inversion
│   │   ├── graph.py        ← Swarm graph builder (adjacency, features)
│   │   ├── gnn.py          ← PyTorch Geometric GNN classifier
│   │   └── behavior.py     ← Behavior classification (6 classes)
│   ├── agents/
│   │   ├── tracker_agent.py   ← Tier-1: individual drone tracking agent
│   │   ├── swarm_agent.py     ← Tier-2: swarm intelligence agent
│   │   └── coordinator.py    ← Tier-3: PPO response coordinator
│   └── simulation/
│       ├── drone.py           ← simulated drone with Reynolds physics
│       ├── swarm_sim.py       ← multi-swarm scenario generator
│       └── scenarios.py       ← named demo scenarios
│
├── services/               ← runnable microservices (each is a process)
│   ├── ingest/             ← sensor data ingestion (RF, acoustic, vision)
│   ├── tracker/            ← Tier-1 agent pool manager
│   ├── swarm_intel/        ← Tier-2 swarm intelligence service
│   ├── coordinator/        ← Tier-3 response coordinator
│   └── gateway/            ← FastAPI WebSocket gateway (UI ↔ services)
│
├── ui/                     ← Streamlit frontend
│   ├── app.py              ← main entry point
│   ├── pages/
│   │   ├── 01_live_ops.py       ← real-time tactical display
│   │   ├── 02_swarm_graph.py    ← swarm topology graph viewer
│   │   ├── 03_behavior.py       ← behavior classification dashboard
│   │   ├── 04_scenarios.py      ← demo scenario launcher
│   │   └── 05_analytics.py      ← historical performance analytics
│   └── components/
│       ├── radar_map.py         ← Plotly radar/map canvas
│       ├── swarm_topology.py    ← PyVis / Plotly graph viz
│       ├── threat_panel.py      ← threat score cards
│       └── track_table.py       ← live track data table
│
├── infra/
│   ├── docker/             ← per-service Dockerfiles
│   ├── k8s/                ← Kubernetes manifests (prod)
│   └── terraform/          ← cloud infra (optional)
│
└── tests/
    ├── unit/               ← pure algorithmic tests
    ├── integration/        ← service-to-service tests
    └── scenarios/          ← end-to-end scenario validation
```

---

## Technology Stack — Authoritative

| Layer | Technology | Version | Notes |
|---|---|---|---|
| Language | Python | 3.11+ | type hints everywhere |
| Frontend | Streamlit | 1.35+ | light theme, custom CSS |
| Visualization | Plotly | 5.20+ | primary charts & radar map |
| Graph Viz | PyVis / Plotly | latest | swarm topology display |
| Tracking | FilterPy | 1.4+ | KF/EKF primitives |
| GNN | PyTorch Geometric | 2.5+ | GNN classifier |
| Deep Learning | PyTorch | 2.2+ | CPU-first, GPU optional |
| RL | Stable-Baselines3 | 2.3+ | PPO coordinator |
| Messaging | ZeroMQ (pyzmq) | 26+ | inter-agent pub/sub |
| API Gateway | FastAPI | 0.110+ | WebSocket + REST |
| Caching | Redis | 7+ | track state, pub/sub |
| Database | PostgreSQL | 16+ | track history, events |
| ORM | SQLAlchemy | 2.0+ | async sessions |
| Task Queue | Celery | 5.3+ | batch GNN inference |
| Simulation | NumPy / SciPy | latest | physics engine |
| RF Sensing | GNU Radio + pyrtlsdr | — | optional hardware |
| Vision | Ultralytics YOLOv8 | 8.1+ | nano model, optional |
| Linting | Ruff + MyPy | latest | strict mode |
| Testing | Pytest + hypothesis | latest | property-based tests |
| Containers | Docker + Compose | — | dev environment |

---

## Coding Standards

### Python Rules
- **Type hints on every function signature** — no exceptions
- **Dataclasses or Pydantic models** for all data structures (no raw dicts in function args)
- **No global mutable state** — agents receive dependencies via constructor injection
- **Async-first** for all I/O (use `asyncio`, `aioredis`, `asyncpg`)
- **NumPy arrays are immutable contracts** — document array shapes in docstrings:
  ```python
  def predict(self, dt: float) -> npt.NDArray[np.float64]:
      """Returns state vector of shape (9,) = [px,py,pz,vx,vy,vz,ax,ay,az]"""
  ```
- **Numerical stability**: always check matrix conditioning before inversion; use
  `np.linalg.solve(A, b)` instead of `np.linalg.inv(A) @ b`
- **Constants** go in `core/constants.py` — no magic numbers in algorithm code

### Mathematical Code Style
```python
# GOOD — mirrors the math paper exactly, with LaTeX reference in docstring
def kalman_update(x: Vec9, P: Mat9x9, z: Vec3, H: Mat3x9, R: Mat3x3) -> tuple[Vec9, Mat9x9]:
    """
    Kalman update step.
    K = P H^T (H P H^T + R)^{-1}
    x = x + K(z - Hx)
    P = (I - KH)P
    """
    S = H @ P @ H.T + R
    K = np.linalg.solve(S.T, (P @ H.T).T).T
    x_new = x + K @ (z - H @ x)
    P_new = (np.eye(len(x)) - K @ H) @ P
    return x_new, P_new

# BAD — opaque, untraceable to math
def update(x, P, z, H, R):
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    return x + K @ (z - H @ x), (np.eye(9) - K @ H) @ P
```

### Streamlit / UI Rules
- **Light color theme only** — see `ui/theme.py` for the design system
- **No `st.experimental_*`** — use stable APIs only
- **All Plotly charts** use the `AEGIS_THEME` dict from `ui/theme.py`
- **Simulation data** is always available via `core/simulation/` — UI never blocks on hardware
- **Interactive demos**: every page must work with simulated data by default;
  real sensor input is an opt-in toggle in the sidebar
- Page layout: `st.set_page_config(layout="wide")` on every page
- Refresh rate: live pages use `st.rerun()` with 500ms sleep; never busy-loop

### Agent Development Rules
- Every agent inherits from `core/agents/base_agent.py`
- Agents communicate **only** via ZMQ messages — no direct function calls between agents
- Agent state must be serializable (MessagePack) for crash recovery
- Each agent publishes heartbeat every 1s to `health` topic

---

## Core Algorithms — Reference Summary

### EKF State Vector
```
x = [px, py, pz, vx, vy, vz, omega]   shape: (7,)
     pos(m) pos(m) pos(m) vel(m/s) x3  turn_rate(rad/s)
```

### Behavior Classes (6)
```python
class BehaviorClass(IntEnum):
    TRANSIT     = 0   # cohesive movement toward target
    ATTACK      = 1   # converging, high speed, low spread
    SCATTER     = 2   # dispersing (saturation maneuver)
    ENCIRCLE    = 3   # pincer / encirclement formation
    DECOY       = 4   # disordered, high separation weight
    UNKNOWN     = 5   # insufficient data
```

### Reynolds Weight Vector → Behavior Mapping
```
[w_sep, w_align, w_coh] → BehaviorClass
[0.3,   0.8,    0.6  ]  → TRANSIT
[0.8,   0.2,    0.1  ]  → SCATTER / ATTACK
[0.5,   0.7,    0.9  ]  → ENCIRCLE
[0.9,   0.1,    0.3  ]  → DECOY
```

### Data Association Cost
```
c_ij = Mahalanobis(z_j, x_i_predicted, S_i)
     = (z_j - H x_i)^T S_i^{-1} (z_j - H x_i)
     gate: reject if c_ij > chi2_threshold(dof=3, p=0.99)
```

---

## Simulation Data Contract

All simulation output follows this schema:

```python
@dataclass
class DroneState:
    drone_id: str          # "swarm_{swarm_id}_drone_{n}"
    swarm_id: str
    timestamp: float       # Unix epoch seconds
    px: float              # position meters (ENU frame)
    py: float
    pz: float
    vx: float              # velocity m/s
    vy: float
    vz: float
    behavior: BehaviorClass
    rf_power_dbm: float    # simulated RF signal
    is_alive: bool

@dataclass
class SwarmState:
    swarm_id: str
    drones: list[DroneState]
    behavior: BehaviorClass
    centroid: npt.NDArray  # shape (3,)
    spread_radius: float
    velocity_coherence: float
    threat_score: float
```

---

## Performance Budgets (NFR Targets)

| Metric | Target | Hard Limit |
|---|---|---|
| EKF cycle time (per drone) | < 1ms | < 2ms |
| Hungarian assignment (100 drones) | < 5ms | < 10ms |
| GNN inference (50-node swarm) | < 20ms | < 50ms |
| End-to-end sensor→threat score | < 100ms | < 200ms |
| Streamlit UI refresh | 500ms | 1000ms |
| Max concurrent drones tracked | 500 | — |
| Max concurrent swarms | 20 | — |
| Track continuity rate | > 95% | > 90% |

---

## Git Workflow
- Branch: `feature/`, `fix/`, `chore/` prefixes
- Commits: Conventional Commits format (`feat:`, `fix:`, `perf:`, `test:`)
- PRs require: passing tests + mypy + ruff + at least one scenario test passing
- Never commit: model weights (use DVC), secrets (use `.env`), large numpy arrays

## Environment Variables
```bash
# .env (copy from .env.example)
AEGIS_ENV=development          # development | staging | production
REDIS_URL=redis://localhost:6379/0
DATABASE_URL=postgresql+asyncpg://aegis:aegis@localhost:5432/aegis
ZMQ_TRACKER_PORT=5555
ZMQ_SWARM_PORT=5556
ZMQ_COORD_PORT=5557
GNN_MODEL_PATH=models/swarm_gnn_v1.pt
SIM_SCENARIO=saturation_attack  # default demo scenario
STREAMLIT_THEME=light
LOG_LEVEL=INFO
```
