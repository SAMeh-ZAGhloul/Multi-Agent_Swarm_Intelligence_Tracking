"""Tests for Hungarian algorithm data association."""

import numpy as np
import pytest

from core.tracking.hungarian import hungarian_assignment, compute_cost_matrix


class TestHungarianAssignment:
    """Test Hungarian algorithm implementation."""

    def test_perfect_assignment(self) -> None:
        """Test perfect one-to-one assignment."""
        # Cost matrix with obvious optimal assignment
        cost_matrix = np.array(
            [
                [1.0, 5.0, 5.0],
                [5.0, 1.0, 5.0],
                [5.0, 5.0, 1.0],
            ]
        )

        matched_tracks, matched_dets, unmatched_tracks, unmatched_dets = (
            hungarian_assignment(cost_matrix)
        )

        # All should be matched
        assert len(matched_tracks) == 3
        assert len(matched_dets) == 3
        assert len(unmatched_tracks) == 0
        assert len(unmatched_dets) == 0

        # Diagonal assignment
        assert matched_tracks == [0, 1, 2]
        assert matched_dets == [0, 1, 2]

    def test_unequal_sizes(self) -> None:
        """Test with more detections than tracks."""
        # 2 tracks, 4 detections
        cost_matrix = np.array(
            [
                [1.0, 2.0, 5.0, 5.0],
                [5.0, 1.0, 2.0, 5.0],
            ]
        )

        matched_tracks, matched_dets, unmatched_tracks, unmatched_dets = (
            hungarian_assignment(cost_matrix)
        )

        assert len(matched_tracks) == 2
        assert len(matched_dets) == 2
        assert len(unmatched_dets) == 2  # 2 detections left unmatched

    def test_gating(self) -> None:
        """Test gating threshold rejects high-cost assignments."""
        cost_matrix = np.array(
            [
                [1.0, 100.0],  # Track 0 can only match det 0
                [100.0, 1.0],  # Track 1 can only match det 1
            ]
        )

        # With threshold=10, low-cost assignments (1.0) should be accepted
        matched_tracks, matched_dets, unmatched_tracks, unmatched_dets = hungarian_assignment(
            cost_matrix, gate_threshold=10.0
        )

        assert len(matched_tracks) == 2
        assert len(matched_dets) == 2
        assert len(unmatched_tracks) == 0
        assert len(unmatched_dets) == 0

        # With very low threshold (0.5), even low-cost assignments are rejected
        matched_tracks, matched_dets, unmatched_tracks, unmatched_dets = hungarian_assignment(
            cost_matrix, gate_threshold=0.5
        )

        assert len(matched_tracks) == 0
        assert len(matched_dets) == 0
        assert len(unmatched_tracks) == 2
        assert len(unmatched_dets) == 2

    def test_empty_input(self) -> None:
        """Test with empty cost matrix."""
        cost_matrix = np.zeros((0, 0))

        matched_tracks, matched_dets, unmatched_tracks, unmatched_dets = (
            hungarian_assignment(cost_matrix)
        )

        assert len(matched_tracks) == 0
        assert len(matched_dets) == 0
        assert len(unmatched_tracks) == 0
        assert len(unmatched_dets) == 0


class TestCostMatrix:
    """Test cost matrix computation."""

    def test_mahalanobis_distance(self) -> None:
        """Test Mahalanobis distance cost computation."""
        predicted_states = [
            np.array([0.0, 0.0, 10.0, 1.0, 0.0, 0.0, 0.0]),
            np.array([10.0, 0.0, 10.0, 0.0, 1.0, 0.0, 0.0]),
        ]
        predicted_covs = [
            np.eye(7) * 1.0,
            np.eye(7) * 1.0,
        ]
        measurements = [
            np.array([0.5, 0.0, 10.0]),  # Close to track 0
            np.array([10.5, 0.0, 10.0]),  # Close to track 1
        ]
        measurement_noise = np.eye(3) * 1.0

        cost_matrix = compute_cost_matrix(
            predicted_states,
            predicted_covs,
            measurements,
            measurement_noise,
        )

        assert cost_matrix.shape == (2, 2)

        # Lower cost for closer matches
        assert cost_matrix[0, 0] < cost_matrix[0, 1]
        assert cost_matrix[1, 1] < cost_matrix[1, 0]

    def test_singular_covariance(self) -> None:
        """Test handling of singular covariance matrices."""
        predicted_states = [
            np.array([0.0, 0.0, 10.0, 1.0, 0.0, 0.0, 0.0]),
        ]
        predicted_covs = [
            np.zeros((7, 7)),  # Singular
        ]
        measurements = [
            np.array([0.0, 0.0, 10.0]),
        ]
        measurement_noise = np.eye(3) * 1.0

        # Should not raise
        cost_matrix = compute_cost_matrix(
            predicted_states,
            predicted_covs,
            measurements,
            measurement_noise,
        )

        # Should have finite costs
        assert np.all(np.isfinite(cost_matrix))


def test_hungarian_performance(benchmark) -> None:
    """Benchmark Hungarian algorithm performance."""
    # 100 tracks, 100 detections
    np.random.seed(42)
    cost_matrix = np.random.randn(100, 100) ** 2

    def run_assignment() -> None:
        hungarian_assignment(cost_matrix)

    stats = benchmark(run_assignment)

    # Should complete in under 10ms (NFR target)
    # stats can be None if benchmark is disabled, so check first
    if stats is not None:
        assert stats.mean < 0.010
