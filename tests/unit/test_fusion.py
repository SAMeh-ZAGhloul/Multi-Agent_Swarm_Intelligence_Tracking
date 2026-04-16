"""Tests for Covariance Intersection fusion."""

import numpy as np
import pytest

from core.tracking.fusion import (
    covariance_intersection,
    find_optimal_omega,
    multi_sensor_fusion,
)


class TestCovarianceIntersection:
    """Test CI fusion implementation."""

    def test_equal_weights(self) -> None:
        """Test fusion with equal weights."""
        x1 = np.array([1.0, 2.0, 3.0])
        P1 = np.eye(3) * 1.0

        x2 = np.array([2.0, 4.0, 6.0])
        P2 = np.eye(3) * 1.0

        x_fused, P_fused = covariance_intersection(x1, P1, x2, P2, omega=0.5)

        # With equal covariances and equal weights, state should be average
        expected = (x1 + x2) / 2
        assert np.allclose(x_fused, expected)

        # With equal covariances, fused covariance equals individual (not smaller)
        # CI guarantees no overconfidence, but doesn't improve with equal info
        assert np.allclose(P_fused, P1)
        assert np.allclose(P_fused, P2)

    def test_unequal_weights(self) -> None:
        """Test fusion with unequal weights."""
        x1 = np.array([0.0, 0.0, 0.0])
        P1 = np.eye(3) * 1.0

        x2 = np.array([10.0, 10.0, 10.0])
        P2 = np.eye(3) * 1.0

        # Weight heavily toward x1
        x_fused, _ = covariance_intersection(x1, P1, x2, P2, omega=0.9)

        # Result should be closer to x1
        assert np.allclose(x_fused, [1.0, 1.0, 1.0], atol=0.5)

    def test_unequal_covariances(self) -> None:
        """Test fusion with different covariance sizes."""
        x1 = np.array([0.0, 0.0, 0.0])
        P1 = np.eye(3) * 0.1  # High certainty

        x2 = np.array([10.0, 10.0, 10.0])
        P2 = np.eye(3) * 10.0  # Low certainty

        # Equal weights
        x_fused, P_fused = covariance_intersection(x1, P1, x2, P2, omega=0.5)

        # Result should be closer to x1 (more certain)
        # Note: CI doesn't automatically weight by certainty
        # The omega parameter controls the weighting
        assert np.linalg.norm(x_fused - x1) < np.linalg.norm(x2 - x1)

    def test_consistency_guarantee(self) -> None:
        """Test that CI maintains consistency (no double counting)."""
        x1 = np.array([1.0, 2.0, 3.0])
        P1 = np.eye(3) * 1.0

        x2 = np.array([1.0, 2.0, 3.0])  # Same state
        P2 = np.eye(3) * 1.0  # Same covariance

        # Even with identical estimates, CI shouldn't over-converge
        x_fused, P_fused = covariance_intersection(x1, P1, x2, P2, omega=0.5)

        # Fused covariance should be positive semi-definite (not overconfident)
        eigenvalues = np.linalg.eigvals(P_fused)
        assert np.all(eigenvalues >= -1e-10)  # Allow tiny numerical noise

        # With identical inputs and equal weights, output equals input
        assert np.allclose(P_fused, P1)

        # Minimum eigenvalue should be positive (covariance is valid)
        min_eig = np.min(eigenvalues)
        assert min_eig > -1e-10

    def test_omega_bounds(self) -> None:
        """Test omega clamping to valid range."""
        x1 = np.array([0.0, 0.0, 0.0])
        P1 = np.eye(3)

        x2 = np.array([1.0, 1.0, 1.0])
        P2 = np.eye(3)

        # Omega < 0 should clamp to 0
        x_fused_low, _ = covariance_intersection(x1, P1, x2, P2, omega=-0.5)
        x_fused_zero, _ = covariance_intersection(x1, P1, x2, P2, omega=0.0)
        assert np.allclose(x_fused_low, x_fused_zero)

        # Omega > 1 should clamp to 1
        x_fused_high, _ = covariance_intersection(x1, P1, x2, P2, omega=1.5)
        x_fused_one, _ = covariance_intersection(x1, P1, x2, P2, omega=1.0)
        assert np.allclose(x_fused_high, x_fused_one)

    def test_singular_covariance_fallback(self) -> None:
        """Test handling of singular covariance matrices."""
        x1 = np.array([1.0, 2.0, 3.0])
        P1 = np.zeros((3, 3))  # Singular

        x2 = np.array([2.0, 4.0, 6.0])
        P2 = np.eye(3)  # Non-singular

        # Should fall back to non-singular estimate
        x_fused, P_fused = covariance_intersection(x1, P1, x2, P2)

        assert np.allclose(x_fused, x2)
        assert np.allclose(P_fused, P2)


class TestOptimalOmega:
    """Test optimal omega finding."""

    def test_find_optimal_trace(self) -> None:
        """Test finding optimal omega by trace minimization."""
        P1 = np.eye(3) * 1.0
        P2 = np.eye(3) * 2.0

        omega = find_optimal_omega(P1, P2, criterion="trace")

        # Should find a valid omega
        assert 0.0 <= omega <= 1.0

    def test_find_optimal_det(self) -> None:
        """Test finding optimal omega by determinant minimization."""
        P1 = np.eye(3) * 1.0
        P2 = np.eye(3) * 0.5

        omega = find_optimal_omega(P1, P2, criterion="det")

        assert 0.0 <= omega <= 1.0


class TestMultiSensorFusion:
    """Test multi-sensor fusion."""

    def test_two_sensors(self) -> None:
        """Test fusion of two sensors."""
        estimates = [
            (np.array([0.0, 0.0, 0.0]), np.eye(3) * 1.0),
            (np.array([2.0, 2.0, 2.0]), np.eye(3) * 1.0),
        ]

        x_fused, P_fused = multi_sensor_fusion(estimates)

        # Should be average
        assert np.allclose(x_fused, [1.0, 1.0, 1.0])

    def test_three_sensors(self) -> None:
        """Test fusion of three sensors."""
        estimates = [
            (np.array([0.0, 0.0, 0.0]), np.eye(3) * 1.0),
            (np.array([3.0, 0.0, 0.0]), np.eye(3) * 1.0),
            (np.array([0.0, 3.0, 0.0]), np.eye(3) * 1.0),
        ]

        x_fused, P_fused = multi_sensor_fusion(estimates)

        # Should be average
        assert np.allclose(x_fused, [1.0, 1.0, 0.0], atol=0.1)

    def test_weighted_fusion(self) -> None:
        """Test fusion with custom weights."""
        estimates = [
            (np.array([0.0, 0.0, 0.0]), np.eye(3) * 1.0),
            (np.array([10.0, 10.0, 10.0]), np.eye(3) * 1.0),
        ]
        weights = [0.9, 0.1]

        x_fused, _ = multi_sensor_fusion(estimates, weights)

        # Should be closer to first estimate (higher weight)
        assert np.linalg.norm(x_fused) < 5.0

    def test_empty_input(self) -> None:
        """Test error on empty input."""
        with pytest.raises(ValueError, match="At least one estimate"):
            multi_sensor_fusion([])

    def test_single_estimate(self) -> None:
        """Test with single estimate (no fusion)."""
        x = np.array([1.0, 2.0, 3.0])
        P = np.eye(3) * 2.0

        x_fused, P_fused = multi_sensor_fusion([(x, P)])

        assert np.allclose(x_fused, x)
        assert np.allclose(P_fused, P)
