"""Behavior classification for drone swarms."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..constants import BehaviorClass

Vec3 = npt.NDArray[np.float64]


class BehaviorClassifier:
    """
    Classify swarm behavior based on kinematic and graph features.

    This classifier uses rule-based heuristics to determine swarm behavior.
    In production, this would be replaced by a GNN model.

    Behaviors
    ---------
    TRANSIT   : Cohesive movement toward target
    ATTACK    : Converging, high speed, low spread
    SCATTER   : Dispersing (saturation maneuver)
    ENCIRCLE  : Pincer / encirclement formation
    DECOY     : Disordered, high separation weight
    UNKNOWN   : Insufficient data
    """

    def __init__(self) -> None:
        """Initialize the behavior classifier."""
        self.last_behavior = BehaviorClass.UNKNOWN
        self.confidence = 0.0

    def classify(
        self,
        positions: npt.NDArray[np.float64],
        velocities: npt.NDArray[np.float64],
        centroid_velocity: Vec3 | None = None,
        spread_radius: float | None = None,
        velocity_coherence: float | None = None,
    ) -> BehaviorClass:
        """
        Classify swarm behavior from kinematic features.

        Parameters
        ----------
        positions : npt.NDArray
            Drone positions, shape (n_drones, 3).
        velocities : npt.NDArray
            Drone velocities, shape (n_drones, 3).
        centroid_velocity : Vec3 | None
            Velocity of swarm centroid. Computed if None.
        spread_radius : float | None
            Average distance from centroid. Computed if None.
        velocity_coherence : float | None
            Mean cosine similarity of velocities. Computed if None.

        Returns
        -------
        BehaviorClass
            Classified behavior.
        """
        n_drones = positions.shape[0]
        if n_drones < 2:
            self.last_behavior = BehaviorClass.UNKNOWN
            self.confidence = 0.0
            return BehaviorClass.UNKNOWN

        # Compute features if not provided
        if centroid_velocity is None:
            centroid_velocity = np.mean(velocities, axis=0)

        if spread_radius is None:
            centroid = np.mean(positions, axis=0)
            distances = np.linalg.norm(positions - centroid, axis=1)
            spread_radius = float(np.mean(distances))

        if velocity_coherence is None:
            speed = np.linalg.norm(velocities, axis=1)
            if np.all(speed > 1e-6):
                normalized_vel = velocities / speed[:, np.newaxis]
                centroid_vel_norm = np.linalg.norm(centroid_velocity)
                if centroid_vel_norm > 1e-6:
                    dot_products = normalized_vel @ centroid_velocity
                    velocity_coherence = float(np.mean(dot_products) / centroid_vel_norm)
                else:
                    velocity_coherence = 0.0
            else:
                velocity_coherence = 0.0

        # Compute mean speed
        mean_speed = float(np.mean(np.linalg.norm(velocities, axis=1)))

        # Compute spread rate (change in spread over time would be here)
        # For single snapshot, use inverse of spread as proxy for "tightness"
        tightness = 1.0 / (spread_radius + 1.0)

        # Classification logic based on Reynolds weight inversion paper
        behavior = self._classify_heuristic(
            velocity_coherence=velocity_coherence,
            mean_speed=mean_speed,
            tightness=tightness,
            spread_radius=spread_radius,
        )

        self.last_behavior = behavior
        return behavior

    def _classify_heuristic(
        self,
        velocity_coherence: float,
        mean_speed: float,
        tightness: float,
        spread_radius: float,
    ) -> BehaviorClass:
        """
        Rule-based behavior classification.

        Behavior heuristics:
        - TRANSIT: high coherence, moderate speed, moderate spread
        - ATTACK: high coherence, high speed, low spread
        - SCATTER: low coherence, high speed, increasing spread
        - ENCIRCLE: moderate coherence, moderate speed, circular pattern
        - DECOY: low coherence, variable speed, high spread
        """
        # Speed thresholds (m/s)
        LOW_SPEED = 5.0
        HIGH_SPEED = 12.0

        # Coherence thresholds
        LOW_COHERENCE = 0.3
        HIGH_COHERENCE = 0.7

        # Spread thresholds (meters)
        TIGHT_SPREAD = 15.0
        WIDE_SPREAD = 40.0

        # ATTACK: high speed, tight formation, high coherence
        if mean_speed > HIGH_SPEED and spread_radius < TIGHT_SPREAD and velocity_coherence > HIGH_COHERENCE:
            self.confidence = 0.85
            return BehaviorClass.ATTACK

        # SCATTER: high speed, low coherence (diverging)
        if mean_speed > HIGH_SPEED and velocity_coherence < LOW_COHERENCE:
            self.confidence = 0.75
            return BehaviorClass.SCATTER

        # ENCIRCLE: moderate coherence, organized pattern
        if (
            LOW_COHERENCE < velocity_coherence < HIGH_COHERENCE
            and LOW_SPEED < mean_speed < HIGH_SPEED
            and spread_radius > WIDE_SPREAD
        ):
            self.confidence = 0.70
            return BehaviorClass.ENCIRCLE

        # DECOY: disordered, low coherence
        if velocity_coherence < LOW_COHERENCE and spread_radius > WIDE_SPREAD:
            self.confidence = 0.65
            return BehaviorClass.DECOY

        # TRANSIT: default for organized movement
        if velocity_coherence > HIGH_COHERENCE:
            self.confidence = 0.80
            return BehaviorClass.TRANSIT

        # UNKNOWN: insufficient pattern
        self.confidence = 0.5
        return BehaviorClass.UNKNOWN

    def classify_batch(
        self,
        swarm_data: list[dict],
    ) -> list[tuple[BehaviorClass, float]]:
        """
        Classify behavior for multiple swarms.

        Parameters
        ----------
        swarm_data : list[dict]
            List of swarm dictionaries with 'positions' and 'velocities' keys.

        Returns
        -------
        list[tuple[BehaviorClass, float]]
            List of (behavior, confidence) tuples.
        """
        results = []
        for swarm in swarm_data:
            positions = np.asarray(swarm["positions"])
            velocities = np.asarray(swarm["velocities"])
            behavior = self.classify(positions, velocities)
            results.append((behavior, self.confidence))
        return results


def behavior_to_threat_score(behavior: BehaviorClass) -> float:
    """
    Convert behavior classification to threat score.

    Parameters
    ----------
    behavior : BehaviorClass
        Classified behavior.

    Returns
    -------
    float
        Threat score in [0, 1].
    """
    threat_mapping = {
        BehaviorClass.ATTACK: 0.95,
        BehaviorClass.SCATTER: 0.80,  # saturation attack
        BehaviorClass.ENCIRCLE: 0.85,
        BehaviorClass.DECOY: 0.40,
        BehaviorClass.TRANSIT: 0.50,
        BehaviorClass.UNKNOWN: 0.30,
    }
    return threat_mapping.get(behavior, 0.30)
