"""Track class for maintaining target state and lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from ..constants import (
    CONFIRMATION_THRESHOLD,
    DELETION_THRESHOLD,
    STATE_DIM,
)
from .ekf import ExtendedKalmanFilter

Vec7 = npt.NDArray[np.float64]
Vec3 = npt.NDArray[np.float64]
Mat7x7 = npt.NDArray[np.float64]


class TrackState(IntEnum):
    """Lifecycle state of a track."""

    TENTATIVE = 0  # newly detected, not yet confirmed
    CONFIRMED = 1  # validated track, ready for association
    COASTING = 2  # temporarily lost, predicting only
    TERMINATED = 3  # track deleted


@dataclass
class Track:
    """
    Represents a single tracked target (drone).

    Attributes
    ----------
    track_id : str
        Unique identifier for this track.
    ekf : ExtendedKalmanFilter
        Kalman filter for state estimation.
    state : TrackState
        Current lifecycle state.
    age : int
        Number of frames since track creation.
    hits : int
        Number of successful associations (detections received).
    misses : int
        Number of consecutive missed associations.
    last_update_time : float
        Unix timestamp of last measurement update.
    """

    track_id: str
    ekf: ExtendedKalmanFilter
    state: TrackState = TrackState.TENTATIVE
    age: int = 0
    hits: int = 0
    misses: int = 0
    last_update_time: float = 0.0

    @classmethod
    def initialize(
        cls,
        track_id: str,
        measurement: Vec3,
        timestamp: float,
        P0: Mat7x7 | None = None,
    ) -> Track:
        """
        Create a new track from an initial measurement.

        Parameters
        ----------
        track_id : str
            Unique track identifier.
        measurement : Vec3
            Initial position measurement [px, py, pz].
        timestamp : float
            Unix timestamp of the measurement.
        P0 : Mat7x7 | None
            Initial covariance. Defaults to large uncertainty.

        Returns
        -------
        Track
            New track instance.
        """
        # Initialize state with position from measurement, zero velocity
        x0 = np.zeros(STATE_DIM, dtype=np.float64)
        x0[:3] = measurement  # position from measurement
        # velocity and omega start at zero

        # Large initial uncertainty
        if P0 is None:
            P0 = np.eye(STATE_DIM, dtype=np.float64) * 100.0

        ekf = ExtendedKalmanFilter(x0=x0, P0=P0)

        return cls(
            track_id=track_id,
            ekf=ekf,
            state=TrackState.TENTATIVE,
            age=1,
            hits=1,
            misses=0,
            last_update_time=timestamp,
        )

    def predict(self, dt: float) -> tuple[Vec7, Mat7x7]:
        """
        Predict track state forward in time.

        Parameters
        ----------
        dt : float
            Time step in seconds.

        Returns
        -------
        x : Vec7
            Predicted state vector.
        P : Mat7x7
            Predicted covariance matrix.
        """
        return self.ekf.predict(dt)

    def update(
        self, measurement: Vec3, timestamp: float
    ) -> tuple[Vec7, Mat7x7, float]:
        """
        Update track with a new measurement.

        Parameters
        ----------
        measurement : Vec3
            New position measurement [px, py, pz].
        timestamp : float
            Unix timestamp of the measurement.

        Returns
        -------
        x : Vec7
            Updated state vector.
        P : Mat7x7
            Updated covariance matrix.
        innovation_norm : float
            Innovation norm (for gating validation).
        """
        self.last_update_time = timestamp
        self.hits += 1
        self.misses = 0

        # Update EKF
        x, P, innovation_norm = self.ekf.update(measurement)

        # Transition state based on hits
        if self.state == TrackState.TENTATIVE and self.hits >= CONFIRMATION_THRESHOLD:
            self.state = TrackState.CONFIRMED

        return x, P, innovation_norm

    def mark_miss(self) -> None:
        """Record a missed association and update state."""
        self.age += 1
        self.misses += 1

        # Update state based on misses
        if self.state == TrackState.CONFIRMED and self.misses > 0:
            self.state = TrackState.COASTING

        # Check for deletion
        if self.misses >= DELETION_THRESHOLD:
            self.state = TrackState.TERMINATED

    def get_position(self) -> Vec3:
        """Get current estimated position."""
        return self.ekf.x[:3].copy()

    def get_velocity(self) -> Vec3:
        """Get current estimated velocity."""
        return self.ekf.x[3:6].copy()

    def get_turn_rate(self) -> float:
        """Get current estimated turn rate (omega)."""
        return float(self.ekf.x[6])

    def get_speed(self) -> float:
        """Get current estimated speed (magnitude of velocity)."""
        return float(np.linalg.norm(self.get_velocity()))

    def is_confirmed(self) -> bool:
        """Check if track is confirmed."""
        return self.state == TrackState.CONFIRMED

    def is_terminated(self) -> bool:
        """Check if track should be deleted."""
        return self.state == TrackState.TERMINATED

    def to_dict(self) -> dict:
        """Serialize track to dictionary for messaging/storage."""
        return {
            "track_id": self.track_id,
            "state": self.state.name,
            "position": self.get_position().tolist(),
            "velocity": self.get_velocity().tolist(),
            "turn_rate": self.get_turn_rate(),
            "speed": self.get_speed(),
            "age": self.age,
            "hits": self.hits,
            "misses": self.misses,
            "confirmed": self.is_confirmed(),
        }
