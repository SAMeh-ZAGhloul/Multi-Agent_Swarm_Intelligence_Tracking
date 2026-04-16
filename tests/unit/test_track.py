"""Tests for Track class."""

import numpy as np
import pytest

from core.tracking.track import Track, TrackState
from core.constants import CONFIRMATION_THRESHOLD, DELETION_THRESHOLD


class TestTrack:
    """Test Track implementation."""

    def test_initialize(self) -> None:
        """Test track initialization."""
        measurement = np.array([10.0, 20.0, 5.0], dtype=np.float64)
        timestamp = 1000.0

        track = Track.initialize("track_001", measurement, timestamp)

        assert track.track_id == "track_001"
        assert track.state == TrackState.TENTATIVE
        assert track.age == 1
        assert track.hits == 1
        assert track.misses == 0

        # Position should match measurement
        assert np.allclose(track.get_position(), measurement)

        # Velocity should be zero initially
        assert np.allclose(track.get_velocity(), [0, 0, 0])

    def test_confirm_track(self) -> None:
        """Test track confirmation after enough hits."""
        measurement = np.array([10.0, 20.0, 5.0], dtype=np.float64)
        track = Track.initialize("track_001", measurement, 1000.0)

        # Should start as tentative
        assert track.state == TrackState.TENTATIVE
        assert not track.is_confirmed()

        # Update until confirmed
        for i in range(CONFIRMATION_THRESHOLD):
            track.update(measurement, 1000.0 + i)

        # Should be confirmed
        assert track.state == TrackState.CONFIRMED
        assert track.is_confirmed()

    def test_track_misses(self) -> None:
        """Test track state transitions on misses."""
        measurement = np.array([10.0, 20.0, 5.0], dtype=np.float64)
        track = Track.initialize("track_001", measurement, 1000.0)

        # Confirm the track first
        for _ in range(CONFIRMATION_THRESHOLD):
            track.update(measurement, 1000.0)

        assert track.state == TrackState.CONFIRMED

        # Add misses
        for i in range(DELETION_THRESHOLD):
            track.mark_miss()

            if i < DELETION_THRESHOLD - 1:
                assert track.state == TrackState.COASTING
                assert not track.is_terminated()

        # Should be terminated
        assert track.state == TrackState.TERMINATED
        assert track.is_terminated()

    def test_state_transitions(self) -> None:
        """Test track state machine transitions."""
        measurement = np.array([10.0, 20.0, 5.0], dtype=np.float64)
        track = Track.initialize("track_001", measurement, 1000.0)

        # TENTATIVE -> CONFIRMED (on hits)
        for _ in range(CONFIRMATION_THRESHOLD):
            track.update(measurement, 1000.0)
        assert track.state == TrackState.CONFIRMED

        # CONFIRMED -> COASTING (on first miss)
        track.mark_miss()
        assert track.state == TrackState.COASTING

        # COASTING -> TERMINATED (on enough misses)
        for _ in range(DELETION_THRESHOLD - 1):
            track.mark_miss()
        assert track.state == TrackState.TERMINATED

    def test_get_speed(self) -> None:
        """Test speed calculation."""
        measurement = np.array([0.0, 0.0, 10.0], dtype=np.float64)
        track = Track.initialize("track_001", measurement, 1000.0)

        # Initially zero velocity
        assert np.isclose(track.get_speed(), 0.0)

        # Set velocity manually via EKF
        track.ekf.x[3:6] = [3.0, 4.0, 0.0]  # 5 m/s

        assert np.isclose(track.get_speed(), 5.0)

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        measurement = np.array([10.0, 20.0, 5.0], dtype=np.float64)
        track = Track.initialize("track_001", measurement, 1000.0)

        track_dict = track.to_dict()

        assert track_dict["track_id"] == "track_001"
        assert track_dict["state"] == "TENTATIVE"
        assert "position" in track_dict
        assert "velocity" in track_dict
        assert "speed" in track_dict
        assert track_dict["hits"] == 1
        assert track_dict["misses"] == 0
