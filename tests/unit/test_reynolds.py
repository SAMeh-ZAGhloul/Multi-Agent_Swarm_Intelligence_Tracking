"""Tests for Reynolds flocking model."""

import numpy as np
import pytest

from core.swarm.reynolds import (
    ReynoldsFlocking,
    ReynoldsWeights,
    compute_reynolds_forces,
)


class TestReynoldsForces:
    """Test Reynolds force computation."""

    def test_separation_force(self) -> None:
        """Test separation force pushes away from close neighbors."""
        position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Neighbor very close
        neighbor_pos = [np.array([1.0, 0.0, 0.0], dtype=np.float64)]
        neighbor_vel = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]

        weights = ReynoldsWeights(separation=1.0, alignment=0.0, cohesion=0.0)

        forces = compute_reynolds_forces(
            position, velocity, neighbor_pos, neighbor_vel, weights
        )

        # Separation should push away from neighbor (negative x direction)
        assert forces.separation[0] < 0

    def test_alignment_force(self) -> None:
        """Test alignment force matches neighbor velocity."""
        position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Neighbor with velocity
        neighbor_pos = [np.array([10.0, 10.0, 10.0], dtype=np.float64)]
        neighbor_vel = [np.array([5.0, 0.0, 0.0], dtype=np.float64)]

        weights = ReynoldsWeights(separation=0.0, alignment=1.0, cohesion=0.0)

        forces = compute_reynolds_forces(
            position, velocity, neighbor_pos, neighbor_vel, weights
        )

        # Alignment should match neighbor velocity direction
        assert forces.alignment[0] > 0

    def test_cohesion_force(self) -> None:
        """Test cohesion force steers toward neighbors."""
        position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Neighbor at distance
        neighbor_pos = [np.array([10.0, 0.0, 0.0], dtype=np.float64)]
        neighbor_vel = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]

        weights = ReynoldsWeights(separation=0.0, alignment=0.0, cohesion=1.0)

        forces = compute_reynolds_forces(
            position, velocity, neighbor_pos, neighbor_vel, weights
        )

        # Cohesion should steer toward neighbor (positive x direction)
        assert forces.cohesion[0] > 0

    def test_no_neighbors(self) -> None:
        """Test with no neighbors returns zero forces."""
        position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        forces = compute_reynolds_forces(
            position, velocity, [], [], ReynoldsWeights()
        )

        assert np.allclose(forces.separation, 0)
        assert np.allclose(forces.alignment, 0)
        assert np.allclose(forces.cohesion, 0)
        assert np.allclose(forces.total, 0)


class TestReynoldsFlocking:
    """Test multi-agent flocking simulation."""

    def test_initialization(self) -> None:
        """Test flock initialization."""
        flock = ReynoldsFlocking(n_agents=10)

        assert flock.n_agents == 10
        assert flock.positions.shape == (10, 3)
        assert flock.velocities.shape == (10, 3)

    def test_initialization_with_positions(self) -> None:
        """Test flock initialization with custom positions."""
        positions = np.random.randn(5, 3)
        velocities = np.random.randn(5, 3)

        flock = ReynoldsFlocking(n_agents=5, positions=positions, velocities=velocities)

        assert np.allclose(flock.positions, positions)
        assert np.allclose(flock.velocities, velocities)

    def test_compute_all_forces(self) -> None:
        """Test force computation for all agents."""
        flock = ReynoldsFlocking(n_agents=10)

        forces = flock.compute_all_forces()

        assert forces.shape == (10, 3)
        assert np.all(np.isfinite(forces))

    def test_step(self) -> None:
        """Test simulation step."""
        flock = ReynoldsFlocking(n_agents=5)

        initial_positions = flock.positions.copy()
        initial_velocities = flock.velocities.copy()

        flock.step(dt=0.1)

        # Positions should change based on velocities
        assert not np.allclose(flock.positions, initial_positions)

        # Velocities should change based on forces
        assert not np.allclose(flock.velocities, initial_velocities)

    def test_speed_clamping(self) -> None:
        """Test that speed is clamped to maximum."""
        flock = ReynoldsFlocking(n_agents=5)

        # Set high initial velocity
        flock.velocities = np.array([[100.0, 0.0, 0.0]] * 5)

        flock.step(dt=0.1, max_speed=15.0)

        # All speeds should be <= max_speed
        speeds = np.linalg.norm(flock.velocities, axis=1)
        assert np.all(speeds <= 15.0 + 1e-6)  # Small tolerance for numerical error

    def test_swarm_metrics(self) -> None:
        """Test swarm metrics computation."""
        flock = ReynoldsFlocking(n_agents=10)

        metrics = flock.get_swarm_metrics()

        assert "centroid" in metrics
        assert "spread_radius" in metrics
        assert "velocity_coherence" in metrics
        assert "mean_speed" in metrics

        assert metrics["centroid"].shape == (3,)
        assert metrics["spread_radius"] >= 0
        assert 0 <= metrics["velocity_coherence"] <= 1
        assert metrics["mean_speed"] >= 0


@pytest.mark.benchmark
def test_flocking_performance(benchmark) -> None:
    """Benchmark flocking force computation."""
    flock = ReynoldsFlocking(n_agents=50)

    result = benchmark(flock.compute_all_forces)

    # Should complete reasonably fast
    assert result is not None
