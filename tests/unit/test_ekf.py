"""Tests for Extended Kalman Filter."""

import numpy as np
import pytest

from core.tracking.ekf import ExtendedKalmanFilter
from core.constants import STATE_DIM, MEASUREMENT_DIM


class TestExtendedKalmanFilter:
    """Test EKF implementation."""

    def test_initialization(self) -> None:
        """Test EKF initializes correctly."""
        ekf = ExtendedKalmanFilter()

        assert ekf.x.shape == (STATE_DIM,)
        assert ekf.P.shape == (STATE_DIM, STATE_DIM)
        assert ekf.Q.shape == (STATE_DIM, STATE_DIM)
        assert ekf.R.shape == (MEASUREMENT_DIM, MEASUREMENT_DIM)

        # Default state should be zeros
        assert np.allclose(ekf.x, 0)

        # Covariance should be positive definite
        assert np.all(np.linalg.eigvals(ekf.P) > 0)

    def test_initialization_with_state(self) -> None:
        """Test EKF initializes with provided state."""
        x0 = np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.1], dtype=np.float64)
        P0 = np.eye(STATE_DIM, dtype=np.float64) * 10

        ekf = ExtendedKalmanFilter(x0=x0, P0=P0)

        assert np.allclose(ekf.x, x0)
        assert np.allclose(ekf.P, P0)

    def test_predict_straight_motion(self) -> None:
        """Test prediction with straight motion (omega=0)."""
        x0 = np.array([0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 0.0], dtype=np.float64)
        ekf = ExtendedKalmanFilter(x0=x0)

        dt = 1.0
        x_pred, P_pred = ekf.predict(dt)

        # Position should change by velocity * dt
        assert np.isclose(x_pred[0], 10.0)  # px = 0 + 10*1
        assert np.isclose(x_pred[1], 0.0)  # py unchanged
        assert np.isclose(x_pred[2], 10.0)  # pz unchanged

        # Velocity should be unchanged
        assert np.allclose(x_pred[3:6], [10.0, 0.0, 0.0])

        # Omega should be unchanged
        assert np.isclose(x_pred[6], 0.0)

    def test_predict_turning_motion(self) -> None:
        """Test prediction with turning motion (omega != 0)."""
        x0 = np.array([0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 0.1], dtype=np.float64)
        ekf = ExtendedKalmanFilter(x0=x0)

        dt = 1.0
        x_pred, _ = ekf.predict(dt)

        # Velocity should rotate
        theta = 0.1  # omega * dt
        expected_vx = 10.0 * np.cos(theta)
        expected_vy = 10.0 * np.sin(theta)

        assert np.isclose(x_pred[3], expected_vx, atol=0.1)
        assert np.isclose(x_pred[4], expected_vy, atol=0.1)

    def test_update(self) -> None:
        """Test measurement update."""
        x0 = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        ekf = ExtendedKalmanFilter(x0=x0)

        # Save covariance before update
        P_before = ekf.P.copy()

        # Measurement at new position
        z = np.array([5.0, 0.0, 10.0], dtype=np.float64)

        x_upd, P_upd, innovation = ekf.update(z)

        # State should move toward measurement
        # With default P=I and R=I, Kalman gain is 0.5, so state moves halfway
        assert np.isclose(x_upd[0], 2.5, atol=0.1)  # px updated (halfway to 5.0)
        assert np.isclose(x_upd[1], 0.0, atol=0.1)  # py unchanged
        assert np.isclose(x_upd[2], 10.0, atol=0.1)  # pz unchanged

        # Covariance should decrease (more certain after update)
        assert np.trace(P_upd) < np.trace(P_before)

    def test_predict_update_cycle(self) -> None:
        """Test full predict-update cycle."""
        x0 = np.array([0.0, 0.0, 10.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float64)
        ekf = ExtendedKalmanFilter(x0=x0)

        # Predict
        ekf.predict(0.1)

        # Update with noisy measurement
        z = np.array([0.6, 0.1, 10.0], dtype=np.float64)  # True pos + noise
        ekf.update(z)

        # Filter should track the target
        assert np.isclose(ekf.x[0], 0.6, atol=0.5)
        assert np.isclose(ekf.x[1], 0.1, atol=0.5)

    def test_copy(self) -> None:
        """Test deep copy."""
        x0 = np.array([1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.1], dtype=np.float64)
        ekf = ExtendedKalmanFilter(x0=x0)

        ekf_copy = ekf.copy()

        # Should have same values
        assert np.allclose(ekf_copy.x, ekf.x)
        assert np.allclose(ekf_copy.P, ekf.P)

        # Should be independent
        ekf.x[0] = 999
        assert not np.isclose(ekf_copy.x[0], 999)

    def test_numerical_stability(self) -> None:
        """Test numerical stability with ill-conditioned matrices."""
        # Very small covariance (high certainty)
        P0 = np.eye(STATE_DIM, dtype=np.float64) * 1e-6
        ekf = ExtendedKalmanFilter(P0=P0)

        # Should not raise
        ekf.predict(0.1)
        z = np.array([0.0, 0.0, 10.0], dtype=np.float64)
        ekf.update(z)

        # Result should be finite
        assert np.all(np.isfinite(ekf.x))
        assert np.all(np.isfinite(ekf.P))


def test_ekf_performance(benchmark) -> None:
    """Benchmark EKF cycle time."""
    ekf = ExtendedKalmanFilter()
    z = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    def run_cycle() -> None:
        ekf.predict(0.1)
        ekf.update(z)

    stats = benchmark(run_cycle)

    # Should complete in under 1ms (NFR target)
    # stats can be None if benchmark is disabled, so check first
    if stats is not None:
        assert stats.mean < 0.001  # 1ms
