# SIMULATION.md — Swarm Simulation Engine
# AEGIS-AI Platform

The simulation engine is the heart of the demo. It generates physically plausible drone
swarm trajectories using the Reynolds flocking model, injects configurable sensor noise,
and produces data streams identical in format to real sensor input.

**Goal:** Demo judges should see behavior that looks genuinely threatening and intelligently
coordinated — not random dots moving around.

---

## 1. Physics Engine

### Reynolds Force Model
```python
# core/simulation/drone.py

def compute_velocity_update(drone_i, neighbors, weights):
    """
    v_i(t+1) = w_s * f_sep(i) + w_a * f_align(i) + w_c * f_coh(i) + f_noise
    """
    w_sep, w_align, w_coh = weights

    # Separation: flee from close neighbors
    f_sep = -sum(
        (p_j - p_i) / (||p_j - p_i||² + ε)
        for p_j in neighbors if ||p_j - p_i|| < r_sep
    )

    # Alignment: match neighbor velocities
    f_align = mean(v_j for v_j in neighbor_velocities) - v_i

    # Cohesion: move toward center of mass
    f_coh = mean(p_j for p_j in neighbor_positions) - p_i

    # Speed limiting
    v_new = w_sep*f_sep + w_align*f_align + w_coh*f_coh
    v_new = clip(v_new, max_speed=30.0)  # m/s

    # Altitude stability (keep drones at target altitude)
    f_alt = (target_altitude - pz) * 0.1
    v_new[2] += f_alt

    return v_new
```

### Behavior-Specific Dynamics
```python
BEHAVIOR_WEIGHTS = {
    BehaviorClass.TRANSIT:  ReynoldsWeights(sep=0.30, align=0.80, coh=0.60),
    BehaviorClass.ATTACK:   ReynoldsWeights(sep=0.80, align=0.20, coh=0.10),
    BehaviorClass.SCATTER:  ReynoldsWeights(sep=0.95, align=0.05, coh=0.05),
    BehaviorClass.ENCIRCLE: ReynoldsWeights(sep=0.50, align=0.70, coh=0.90),
    BehaviorClass.DECOY:    ReynoldsWeights(sep=0.90, align=0.10, coh=0.30),
}

# Behavior transitions are smooth (weights interpolated over 5 seconds)
# to avoid discontinuous jumps in the trajectory data
```

---

## 2. Sensor Noise Models

```python
# core/simulation/noise.py

class RFNoiseModel:
    """Simulates RF detection noise."""
    sigma_pos: float = 2.0   # meters (position uncertainty)
    miss_prob: float = 0.02  # 2% chance of missing a drone per frame
    false_alarm_rate: float = 0.005  # false detections per frame per sensor

class AcousticNoiseModel:
    sigma_pos: float = 5.0   # worse position accuracy
    sigma_bearing: float = 3.0  # degrees
    max_range: float = 500.0    # meters

class VisionNoiseModel:
    sigma_px: float = 3.0     # pixels → converted to meters by geometry
    miss_prob_occluded: float = 0.15  # higher miss rate when drones overlap

# All models produce ObservationMessage-compatible output
# The ingest service handles fusion via Covariance Intersection
```

---

## 3. Scenario Definitions

### SCN-01: Single Swarm Transit
```
Duration:    60 seconds
Drones:      20
Swarms:      1
Start pos:   [-1500, 0] (western edge)
Target:      [+1500, 0] (eastern asset)
Phases:
  T=0s:  Formation: loose diamond, TRANSIT behavior
  T=30s: Behavior transition → ATTACK (speed increases, spread decreases)
  T=45s: Drones within 200m of target, threat score → CRITICAL
Expected output:
  - Single swarm tracked throughout
  - Behavior transition detected at T=30s ± 2s
  - Threat alert CRITICAL at T=45s ± 3s
```

### SCN-02: Saturation Attack
```
Duration:    90 seconds
Drones:      50
Swarms:      1
Phases:
  T=0s:   TRANSIT — 50 drones in column formation approaching from north
  T=30s:  SCATTER — swarm disperses laterally (spread_radius doubles in 10s)
  T=50s:  ATTACK  — all 50 drones converge on target from multiple angles
Expected output:
  - Track continuity > 95% through the scatter event
  - Spread radius increase detected (scatter recognized)
  - 50-target saturation → RF jammer recommendation insufficient alone
```

### SCN-03: Pincer Maneuver
```
Duration:    75 seconds
Drones:      30 (15 per swarm)
Swarms:      2
Phases:
  T=0s:  Both swarms in TRANSIT approaching from NE and NW
  T=25s: Swarm-Alpha → ENCIRCLE (clockwise arc)
         Swarm-Beta  → ENCIRCLE (counterclockwise arc)
  T=50s: Both swarms converge — encirclement complete
Expected output:
  - Two separate swarms maintained throughout
  - ENCIRCLE behavior detected on both
  - Threat assessment ranks both equally (same distance, size)
```

### SCN-04: Decoy + Real Attack
```
Duration:    90 seconds
Drones:      40 (25 decoy + 15 real)
Swarms:      2
Swarm-Decoy:  25 drones, DECOY behavior (disordered, high entropy)
Swarm-Attack: 15 drones, TRANSIT then ATTACK
Expected output:
  - Swarm-Attack ranks HIGHER in threat assessment despite fewer drones
    (lower behavior_confidence for decoy + higher approach_rate for attack)
  - Decoy correctly classified as DECOY (not ATTACK)
  - Demonstrates behavior analysis adds value beyond just drone count
```

### SCN-05: Adaptive Swarm (Learning)
```
Duration:    120 seconds
Drones:      60 (3 swarms of 20)
Swarms:      3
Adaptive behavior:
  When any drone is destroyed (threat score causes "neutralization event"):
  - Surviving drones in that swarm adjust behavior
  - Swarm routing changes based on neutralized countermeasure positions
  - Implemented via: when neutralization event received, toggle to SCATTER
    then re-form with new approach vector
Expected output:
  - Multiple behavior transitions per swarm
  - System maintains tracking through 30% drone loss
  - GNN captures formation changes in swarm topology graph
```

---

## 4. Simulation Data Rate and Format

```python
# Simulator outputs at 10Hz (configurable)
# Each tick produces one ObservationMessage per active sensor

# Typical output rate for 50 drones, 3 sensors:
# 50 × 3 sensors × 10Hz = 1500 ObservationMessages/second
# After fusion: 50 × 10Hz = 500 fused observations/second

# Each ObservationMessage is ~300 bytes (MessagePack)
# Total: ~450 KB/s through the ingest pipeline
# Well within Redis stream capacity (typical: 50 MB/s)
```

---

## 5. Ground Truth Logging

```python
# The simulator writes ground truth to PostgreSQL every tick
# This enables post-run accuracy calculation

class GroundTruthRecord(Base):
    __tablename__ = "ground_truth"
    run_id:      str    # links to scenario run
    timestamp:   float
    drone_id:    str
    swarm_id:    str
    true_px:     float; true_py: float; true_pz: float
    true_vx:     float; true_vy: float; true_vz: float
    true_behavior: int  # BehaviorClass

# After scenario completes, analytics service computes:
# - Track-to-truth association (Hungarian on final positions)
# - Position RMSE per track
# - Behavior classification accuracy (compare to true_behavior)
# - Track continuity (% of ground truth drones with continuous tracks)
```

---

## 6. Simulation API (for UI Scenario Launcher)

```python
# core/simulation/swarm_sim.py

class SwarmSimulator:
    def start(self, scenario_id: str, overrides: ScenarioOverrides) -> str:
        """Returns run_id. Starts pushing to ZMQ."""

    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def stop(self) -> None: ...

    @property
    def status(self) -> SimStatus:
        return SimStatus(
            running=..., elapsed_s=..., drone_count=...,
            active_swarms=..., behavior_by_swarm={...}
        )

    def inject_behavior(self, swarm_id: str, behavior: BehaviorClass) -> None:
        """Manual override for demo: change behavior mid-scenario."""
        # Used by the Reynolds parameter sliders in the UI
```

---

## 7. Generating Training Data for GNN

```bash
# scripts/generate_training_data.py
# Generates 50,000 labeled swarm graph snapshots for GNN training

# Run with:
python scripts/generate_training_data.py \
    --n-samples 50000 \
    --output data/training/swarm_graphs.pt \
    --seed 12345

# Distribution across behavior classes:
# TRANSIT:  20% (10,000 samples)
# ATTACK:   20% (10,000 samples)
# SCATTER:  15% (7,500 samples)
# ENCIRCLE: 15% (7,500 samples)
# DECOY:    15% (7,500 samples)
# UNKNOWN:  15% (7,500 samples) — sparse/disconnected graphs

# Each sample:
# - Swarm size: 5–100 drones (uniform random)
# - Noise level: 0–3m sigma (uniform random)
# - Behavior duration: 5–60s slice of a Reynolds trajectory
# - Label: true BehaviorClass from Reynolds weight vector
```
