# TESTING.md — Test Strategy & Coverage
# AEGIS-AI Platform

---

## 1. Testing Philosophy

- **Every algorithm has a mathematical ground truth** — use it. EKF tests compare against
  analytical solutions. Reynolds inversion tests use synthetic data with known weights.
- **Simulation is the test oracle** — since we generate ground truth, we can always measure
  accuracy precisely. Never accept hand-wavy "it looks right" validation.
- **Tests must be deterministic** — always use fixed seeds. Flaky tests are treated as bugs.
- **Performance tests are required** — latency regressions are functional failures.

---

## 2. Test Pyramid

```
                    ┌──────────────┐
                    │  E2E / Demo  │  5 scenario tests
                    │  Scenarios   │  (slow, run before demo)
                  ┌─┴──────────────┴─┐
                  │   Integration    │  ~30 tests
                  │  (services +     │  (Docker required)
                  │   Redis)         │
                ┌─┴──────────────────┴─┐
                │     Unit Tests       │  ~200 tests
                │  (core/ algorithms)  │  (fast, no I/O)
                └──────────────────────┘
```

Target coverage: `core/` → 95%+, `services/` → 80%+, `ui/` → 60%+

---

## 3. Unit Tests

### 3.1 EKF Tests (`tests/unit/test_ekf.py`)
```python
def test_kalman_gain_zero_process_noise():
    """With Q=0, Kalman gain should decrease monotonically."""

def test_kalman_gain_zero_measurement_noise():
    """With R=0, Kalman gain should be 1 (trust measurement fully)."""

def test_ekf_constant_velocity_prediction():
    """Constant velocity drone: EKF position error < 0.01m after 10 steps."""

def test_ekf_circular_motion_ctra():
    """Circular motion at omega=0.1rad/s: RMSE < 0.5m over 100 steps."""

def test_covariance_positive_definite():
    """P matrix must remain positive definite (all eigenvalues > 0) throughout."""

def test_ekf_recovers_from_missed_detections():
    """After 5 consecutive missed detections, error should still be < 10m."""
```

### 3.2 Hungarian Assignment Tests (`tests/unit/test_hungarian.py`)
```python
def test_perfect_assignment_no_noise():
    """With zero noise, each track matches its generating measurement."""

def test_assignment_with_gating():
    """Measurements outside chi2 gate should be rejected (new tracks initiated)."""

def test_lapjv_vs_scipy_equivalence():
    """LAPJV and scipy.optimize.linear_sum_assignment must return same assignment."""

def test_assignment_n_measurements_lt_n_tracks():
    """Handles case where fewer measurements than existing tracks (missed detections)."""

def test_assignment_n_measurements_gt_n_tracks():
    """Handles case where more measurements than tracks (new targets)."""
```

### 3.3 Reynolds Model Tests (`tests/unit/test_reynolds.py`)
```python
def test_inversion_transit_weights():
    """Generate transit swarm with known [0.3,0.8,0.6], invert, error < 0.05."""
    swarm = generate_reynolds_swarm(weights=[0.3, 0.8, 0.6], n=20, seed=42)
    estimated = invert_reynolds_weights(swarm.trajectories)
    assert np.allclose(estimated, [0.3, 0.8, 0.6], atol=0.05)

def test_inversion_attack_weights():
    """Attack swarm [0.8,0.2,0.1] correctly inverted."""

def test_spread_radius_decreases_in_attack():
    """Attack behavior: spread_radius should decrease over 30 steps."""

def test_velocity_coherence_high_in_transit():
    """Transit behavior: velocity_coherence > 0.8."""

def test_formation_entropy_high_in_decoy():
    """Decoy behavior: formation_entropy > 0.9 (disordered)."""
```

### 3.4 GNN Tests (`tests/unit/test_gnn.py`)
```python
def test_gnn_output_shape():
    """Model output shape is (batch_size, 6) for any valid input graph."""

def test_gnn_probabilities_sum_to_one():
    """Softmax output sums to 1.0 for every sample."""

def test_gnn_invariant_to_node_ordering():
    """Permuting node order must not change classification."""

def test_gnn_handles_single_drone():
    """Graph with 1 node → UNKNOWN class with high confidence."""

def test_gnn_handles_disconnected_graph():
    """Swarm with no edges (all drones far apart) should not crash."""

@pytest.mark.slow
def test_gnn_accuracy_on_test_set():
    """Accuracy on 1000-sample held-out set must exceed 88%."""
    # loads pretrained model from GNN_MODEL_PATH
```

### 3.5 Covariance Intersection Tests (`tests/unit/test_fusion.py`)
```python
def test_ci_result_consistent():
    """Fused covariance must be >= individual covariances (no overconfidence)."""

def test_ci_omega_minimizes_trace():
    """omega should minimize tr(P_fused) within [0, 1]."""

def test_ci_single_source():
    """CI fusion with one source should return that source unchanged."""
```

### 3.6 Threat Score Tests (`tests/unit/test_coordinator.py`)
```python
def test_attack_swarm_scores_higher_than_decoy():
    """ATTACK behavior swarm should score > DECOY behavior swarm at same distance."""

def test_closer_swarm_scores_higher():
    """Same behavior, closer swarm gets higher threat score."""

def test_larger_swarm_scores_higher():
    """Same behavior and distance, more drones → higher score."""

def test_ppo_respects_countermeasure_count():
    """With 2 jammers available, coordinator should not assign 3."""
```

---

## 4. Integration Tests

### 4.1 Tracker Service Integration (`tests/integration/test_tracker_service.py`)
```python
@pytest.mark.integration
async def test_tracker_processes_observations():
    """Push 10 observations to Redis, verify 10 TrackStates appear in aegis:tracks."""

@pytest.mark.integration
async def test_track_lifecycle_confirmed():
    """Track reaches CONFIRMED after N_confirm consecutive observations."""

@pytest.mark.integration
async def test_track_lifecycle_deleted():
    """Track reaches DELETED after N_delete consecutive misses."""
```

### 4.2 Swarm Intel Integration (`tests/integration/test_swarm_intel_service.py`)
```python
@pytest.mark.integration
async def test_swarm_grouper_clusters_nearby_tracks():
    """20 closely-spaced tracks grouped into 1 swarm."""

@pytest.mark.integration
async def test_behavior_report_published():
    """SwarmIntelReport appears in aegis:swarm_intel within 2 seconds of tracks."""
```

### 4.3 Gateway WebSocket Integration (`tests/integration/test_gateway.py`)
```python
@pytest.mark.integration
async def test_websocket_snapshot_received():
    """Connect to ws://localhost:8000/ws/live, receive snapshot within 1 second."""

@pytest.mark.integration
async def test_scenario_start_via_rest():
    """POST /scenarios/start returns run_id and simulator starts producing data."""
```

---

## 5. Scenario Tests (E2E)

```python
# tests/scenarios/test_demo_scenarios.py

@pytest.mark.scenario
@pytest.mark.parametrize("scenario_id", ["SCN-01","SCN-02","SCN-03","SCN-04","SCN-05"])
def test_scenario_completes_without_crash(scenario_id):
    """Run scenario to completion. No exceptions."""

@pytest.mark.scenario
def test_scn02_track_continuity():
    """Saturation attack (50 drones): track continuity rate > 95%."""

@pytest.mark.scenario
def test_scn04_decoy_vs_attack_ranking():
    """Decoy swarm ranks LOWER than attack swarm in threat assessment."""

@pytest.mark.scenario
def test_scn05_behavior_transition_detected():
    """Behavior class transition is detected within 2 seconds of simulator change."""
```

---

## 6. Performance / Benchmark Tests

```python
# tests/benchmarks/test_performance.py

@pytest.mark.benchmark
def test_ekf_cycle_time_per_drone(benchmark):
    """EKF predict+update must complete < 1ms per drone."""
    result = benchmark(run_ekf_cycle, drone_count=1)
    assert result.stats.mean < 0.001  # 1ms

@pytest.mark.benchmark
def test_hungarian_100_drones(benchmark):
    """LAPJV assignment for 100 drones < 5ms."""
    result = benchmark(run_hungarian, n=100)
    assert result.stats.mean < 0.005

@pytest.mark.benchmark
def test_gnn_inference_50_nodes(benchmark):
    """GNN inference for 50-node graph < 20ms on CPU."""
    result = benchmark(run_gnn_inference, n_nodes=50)
    assert result.stats.mean < 0.020
```

---

## 7. Test Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "slow: marks tests as slow (run with --slow)",
    "integration: requires running Docker services",
    "scenario: full end-to-end scenario test",
    "benchmark: performance regression test",
]
filterwarnings = ["error::DeprecationWarning"]

[tool.coverage.run]
source = ["core", "services"]
omit = ["*/tests/*", "*/migrations/*"]

[tool.coverage.report]
fail_under = 80
show_missing = true
```

---

## 8. CI Pipeline (GitHub Actions sketch)

```yaml
# .github/workflows/ci.yml
jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - pytest tests/unit/ -v --cov=core --cov-fail-under=90

  lint:
    steps:
      - ruff check .
      - mypy core/ services/ --strict

  integration:
    services: { redis: ..., postgres: ... }
    steps:
      - pytest tests/integration/ -m integration

  benchmark:
    steps:
      - pytest tests/benchmarks/ -m benchmark --benchmark-compare
```

---

## 9. Property-Based Tests (Hypothesis)

```python
# tests/unit/test_ekf_properties.py
from hypothesis import given, strategies as st
import numpy as np

@given(
    pos=st.lists(st.floats(-10000, 10000), min_size=3, max_size=3),
    vel=st.lists(st.floats(-100, 100), min_size=3, max_size=3),
)
def test_ekf_covariance_always_positive_definite(pos, vel):
    """For any valid initial state, P remains PD after 100 steps."""
    ekf = DroneEKF()
    ekf.initialize(pos + vel + [0.0])
    for _ in range(100):
        ekf.predict(dt=0.1)
        eigenvalues = np.linalg.eigvalsh(ekf.P)
        assert np.all(eigenvalues > -1e-10)  # allow tiny numerical noise

@given(n=st.integers(2, 50))
def test_hungarian_assignment_always_valid(n):
    """For any n×n cost matrix, assignment has exactly n pairs."""
    cost = np.random.rand(n, n)
    row_ind, col_ind = lapjv_assignment(cost)
    assert len(row_ind) == n
    assert len(set(col_ind)) == n  # no repeated assignments
```
