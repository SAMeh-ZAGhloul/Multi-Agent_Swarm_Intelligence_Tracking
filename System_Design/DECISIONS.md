# DECISIONS.md — Architecture Decision Records (ADRs)
# AEGIS-AI Platform

ADRs document *why* key decisions were made. When Claude Code is implementing a feature
and wonders "why is it done this way?", the answer is here.

---

## ADR-001: Python as primary language (not C++/Java)

**Decision:** Python 3.11 for all components  
**Status:** Accepted

**Rationale:**
- PyTorch Geometric (GNN), FilterPy (EKF), Stable-Baselines3 (RL) are Python-native
- Competition demo prioritizes development speed over raw performance
- NumPy with Numba JIT achieves adequate EKF throughput (< 1ms/drone)
- Type hints + Pydantic give sufficient safety for the codebase size

**Consequences:**
- EKF must use NumPy vectorization, not naïve Python loops
- GNN inference on CPU is ~15ms — acceptable for demo, GPU needed for production
- Cannot mix C extensions without explicit FFI wrapper

---

## ADR-002: Streamlit (not React/Vue) for frontend

**Decision:** Streamlit with Plotly for all UI  
**Status:** Accepted

**Rationale:**
- Competition audience is technical (military/engineering) — functionality > aesthetics
- Streamlit allows rapid iteration: full page in ~100 lines
- Plotly is battle-tested for real-time map/chart displays
- No front-end build toolchain complexity
- All team members are Python-fluent

**Consequences:**
- Refresh mechanism is `st.rerun()` loop — not true WebSocket push
- Limited fine-grained UI control vs React
- Custom CSS injected via `st.markdown(unsafe_allow_html=True)` for styling

**Alternative rejected:** FastAPI + React SPA — correct long-term choice but 3× development time

---

## ADR-003: Redis Streams (not Kafka) for messaging

**Decision:** Redis Streams for all inter-service message passing  
**Status:** Accepted

**Rationale:**
- Single-node demo doesn't need Kafka's distribution
- Redis already required for caching — reduces infrastructure complexity
- Redis Streams support consumer groups (needed for tracker worker pool)
- < 100ms end-to-end latency requirement: Redis at ~0.1ms vs Kafka at ~5ms
- In-memory → no disk I/O bottleneck

**Consequences:**
- Data not durable across Redis restart (acceptable in demo)
- Max message size soft limit: 1MB (well within our message schema)
- If scaling to Profile C (distributed), Redis Cluster or migration to Kafka

---

## ADR-004: EKF (not Particle Filter or UKF) for drone tracking

**Decision:** Extended Kalman Filter with CTRA motion model  
**Status:** Accepted

**Rationale:**
- Drone dynamics are smooth and differentiable — EKF linearization is valid
- EKF runs in ~0.3ms vs Particle Filter at ~5ms per drone (500 particles)
- Particle filter would require 500 × 500 = 250,000 particles for 500 drones: infeasible
- UKF is more accurate for highly nonlinear dynamics, but CTRA-EKF is sufficient
  for consumer drone maneuverability ranges
- FilterPy library provides battle-tested EKF implementation

**Consequences:**
- EKF degrades for highly agile drones (rapid acceleration changes)
- Mitigation: Interacting Multiple Models (IMM) can be added as future extension
  to handle mode-switching between constant-velocity, constant-acceleration, and CTRA

---

## ADR-005: DBSCAN (not K-means) for swarm grouping

**Decision:** Modified DBSCAN with velocity-weighted distance metric  
**Status:** Accepted

**Rationale:**
- Number of swarms is unknown a priori — DBSCAN doesn't require K
- DBSCAN handles arbitrary swarm shapes (important for ENCIRCLE formation)
- Noise points (isolated drones between swarms) naturally classified as UNKNOWN
- K-means would force every drone into a swarm even if equidistant

**Consequences:**
- DBSCAN parameters (eps, min_samples) must be tuned per scenario
- Default: eps=200m (spatial component), re-run every 2 seconds
- Edge case: single drone separated from swarm may oscillate between swarm membership

---

## ADR-006: GAT (Graph Attention Network) over GCN for GNN

**Decision:** Graph Attention Network (GAT) with 4 attention heads  
**Status:** Accepted

**Rationale:**
- Leader-follower relationships in swarms have unequal importance — attention captures this
- Attention weights α_ij are directly interpretable: high weight = strong leader influence
- GCN averages all neighbors equally — misses the hierarchical swarm structure
- Visualization benefit: attention weights can be rendered as edge thickness in UI

**Consequences:**
- GAT has higher parameter count than GCN (but still < 500K params total)
- Training requires larger dataset to avoid attention head collapse
- Mitigation: multi-head averaging + dropout

---

## ADR-007: PPO (not DQN) for response coordination

**Decision:** Proximal Policy Optimization (PPO) for countermeasure allocation  
**Status:** Accepted

**Rationale:**
- Action space is continuous (resource allocation fractions) — PPO handles this naturally
- DQN requires discretized actions: impractical for 20-swarm × 5-countermeasure space
- PPO's clipped surrogate objective prevents catastrophic policy updates
- Stable-Baselines3 provides well-tested PPO with customizable network architecture
- Trained entirely in simulation — no real environment required

**Consequences:**
- PPO policy must be re-trained if threat score weights change significantly
- Training takes ~2 hours on CPU (acceptable for demo prep, not online learning)
- For v1.0 demo: policy is pre-trained and static (loaded from file)

---

## ADR-008: Covariance Intersection over Naive Bayesian Fusion

**Decision:** Covariance Intersection (CI) for multi-sensor fusion  
**Status:** Accepted

**Rationale:**
- Cross-correlations between RF/acoustic/vision sensors are unknown and time-varying
- Naive Bayesian fusion (direct covariance averaging) produces overconfident estimates
  when sensors share information (e.g., both observing the same distinctive RF signature)
- CI is proven to be consistent (no overconfidence) for any unknown correlation
- Adds ~0.1ms overhead per observation — negligible

**Consequences:**
- CI is conservative — fused covariance is larger than optimal with known correlations
- Mitigation: as more observations arrive, estimates converge regardless
- For single-sensor scenarios (hardware failure), CI degrades to that sensor's estimate

---

## ADR-009: Light color theme throughout

**Decision:** Light theme for all UI, no dark mode  
**Status:** Accepted

**Rationale:**
- Competition demo is presented on projector/large screen in a lit room
- Dark themes lose contrast on projectors (gray-on-black becomes invisible)
- Military operations centers historically use light maps (ICAO standard)
- Behavior class colors designed for light backgrounds (they desaturate on dark)

**Consequences:**
- Must set `base = "light"` in `.streamlit/config.toml`
- All Plotly figures must set `paper_bgcolor="white"` and `plot_bgcolor` to light color
- Custom CSS injection required to override Streamlit's occasional dark elements

---

## ADR-010: Simulation-only for v1.0 (no real hardware)

**Decision:** All v1.0 functionality uses simulated sensor data  
**Status:** Accepted

**Rationale:**
- Competition allows "ideas and creative concepts" — implementation proof, not field system
- Real RTL-SDR + drone hardware introduces demo risk (RF interference, licensing)
- Simulation produces more controllable, repeatable, impressive demonstrations
- Architecture is hardware-agnostic: real sensors plug in as alternative ObservationMessage sources

**Consequences:**
- Simulation ground truth enables precise accuracy measurement (see TESTING.md)
- Must clearly label UI: "SIMULATION MODE" indicator in sidebar
- Hardware integration is a clear Phase 2 roadmap item
