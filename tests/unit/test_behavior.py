"""Tests for behavior classification."""

import numpy as np
import pytest

from core.swarm.behavior import BehaviorClassifier, behavior_to_threat_score
from core.constants import BehaviorClass


class TestBehaviorClassifier:
    """Test behavior classification."""

    def test_empty_input(self) -> None:
        """Test classification with insufficient data."""
        classifier = BehaviorClassifier()

        positions = np.zeros((0, 3))
        velocities = np.zeros((0, 3))

        behavior = classifier.classify(positions, velocities)

        assert behavior == BehaviorClass.UNKNOWN

    def test_single_drone(self) -> None:
        """Test classification with single drone."""
        classifier = BehaviorClassifier()

        positions = np.array([[0.0, 0.0, 10.0]])
        velocities = np.array([[1.0, 0.0, 0.0]])

        behavior = classifier.classify(positions, velocities)

        assert behavior == BehaviorClass.UNKNOWN

    def test_attack_behavior(self) -> None:
        """Test ATTACK behavior detection."""
        classifier = BehaviorClassifier()

        # Tight formation, high speed, coherent
        n_drones = 10
        positions = np.random.randn(n_drones, 3) * 5  # Tight spread
        velocities = np.array([[10.0, 0.0, 0.0]] * n_drones)  # High speed, coherent

        behavior = classifier.classify(positions, velocities)

        # Should detect as ATTACK or similar high-threat
        assert behavior in (BehaviorClass.ATTACK, BehaviorClass.TRANSIT)

    def test_scatter_behavior(self) -> None:
        """Test SCATTER behavior detection."""
        classifier = BehaviorClassifier()

        # Dispersing drones
        n_drones = 10
        positions = np.random.randn(n_drones, 3) * 20  # Wide spread
        # Velocities pointing outward
        velocities = positions * 0.5  # Moving away from center

        behavior = classifier.classify(positions, velocities)

        # May detect as SCATTER or DECOY
        assert behavior in (BehaviorClass.SCATTER, BehaviorClass.DECOY, BehaviorClass.UNKNOWN)

    def test_transit_behavior(self) -> None:
        """Test TRANSIT behavior detection."""
        classifier = BehaviorClassifier()

        # Cohesive movement
        n_drones = 10
        positions = np.random.randn(n_drones, 3) * 15  # Moderate spread
        velocities = np.array([[8.0, 0.0, 0.0]] * n_drones)  # Same velocity

        behavior = classifier.classify(positions, velocities)

        # High coherence should be TRANSIT
        assert behavior == BehaviorClass.TRANSIT

    def test_confidence_tracking(self) -> None:
        """Test that confidence is tracked."""
        classifier = BehaviorClassifier()

        positions = np.random.randn(10, 3) * 10
        velocities = np.random.randn(10, 3)

        classifier.classify(positions, velocities)

        assert 0.0 <= classifier.confidence <= 1.0

    def test_batch_classification(self) -> None:
        """Test batch classification of multiple swarms."""
        classifier = BehaviorClassifier()

        swarm_data = [
            {
                "positions": np.random.randn(10, 3) * 10,
                "velocities": np.random.randn(10, 3),
            }
            for _ in range(3)
        ]

        results = classifier.classify_batch(swarm_data)

        assert len(results) == 3
        for behavior, confidence in results:
            assert isinstance(behavior, BehaviorClass)
            assert 0.0 <= confidence <= 1.0


class TestThreatScore:
    """Test threat score conversion."""

    def test_attack_threat(self) -> None:
        """Test ATTACK threat score."""
        score = behavior_to_threat_score(BehaviorClass.ATTACK)
        assert score > 0.8

    def test_scatter_threat(self) -> None:
        """Test SCATTER threat score."""
        score = behavior_to_threat_score(BehaviorClass.SCATTER)
        assert score > 0.6

    def test_encircle_threat(self) -> None:
        """Test ENCIRCLE threat score."""
        score = behavior_to_threat_score(BehaviorClass.ENCIRCLE)
        assert score > 0.7

    def test_transit_threat(self) -> None:
        """Test TRANSIT threat score."""
        score = behavior_to_threat_score(BehaviorClass.TRANSIT)
        assert 0.3 < score < 0.7

    def test_decoy_threat(self) -> None:
        """Test DECOY threat score."""
        score = behavior_to_threat_score(BehaviorClass.DECOY)
        assert score < 0.6

    def test_unknown_threat(self) -> None:
        """Test UNKNOWN threat score."""
        score = behavior_to_threat_score(BehaviorClass.UNKNOWN)
        assert score < 0.5

    def test_threat_ordering(self) -> None:
        """Test that threat scores are properly ordered."""
        attack_score = behavior_to_threat_score(BehaviorClass.ATTACK)
        decoy_score = behavior_to_threat_score(BehaviorClass.DECOY)
        unknown_score = behavior_to_threat_score(BehaviorClass.UNKNOWN)

        # ATTACK should be highest
        assert attack_score > decoy_score
        assert attack_score > unknown_score
