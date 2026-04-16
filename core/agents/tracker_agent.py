"""Tier-1 Tracker Agent for individual drone tracking."""

from __future__ import annotations

import asyncio
import time
import uuid

import numpy as np
import numpy.typing as npt

from ..constants import GATING_THRESHOLD, MEASUREMENT_NOISE_POS
from ..tracking import Track
from ..tracking.hungarian import compute_cost_matrix, hungarian_assignment
from .base_agent import AgentMessage, BaseAgent

Vec3 = npt.NDArray[np.float64]
Mat3x3 = npt.NDArray[np.float64]


class TrackerAgent(BaseAgent):
    """
    Tier-1 Agent: Individual drone tracking.

    Responsibilities:
    - Maintain EKF tracks for individual drones
    - Associate new detections with existing tracks
    - Create new tracks for unassociated detections
    - Publish track updates to Tier-2 (Swarm Agent)

    Messages Received:
    - sensor/detection: New detection from ingest service
    - tracker/query: Request for specific track data

    Messages Sent:
    - track/update: Updated track states
    - track/new: Newly created tracks
    - track/lost: Tracks that have been terminated
    """

    def __init__(
        self,
        agent_id: str | None = None,
        max_tracks: int = 500,
    ) -> None:
        """
        Initialize the tracker agent.

        Parameters
        ----------
        agent_id : str | None
            Agent identifier. Generated if None.
        max_tracks : int
            Maximum number of concurrent tracks.
        """
        super().__init__(
            agent_id=agent_id or f"tracker_{uuid.uuid4().hex[:8]}",
            agent_type="tracker",
        )
        self.max_tracks = max_tracks
        self.tracks: dict[str, Track] = {}
        self._track_counter = 0
        self._last_update_time: float = 0.0

        # Measurement noise for cost matrix
        self._R = MEASUREMENT_NOISE_POS**2 * np.eye(3, dtype=np.float64)

    def _generate_track_id(self) -> str:
        """Generate unique track ID."""
        self._track_counter += 1
        return f"track_{self.agent_id}_{self._track_counter}"

    async def start(self) -> None:
        """Start the tracker agent main loop."""
        self.running = True
        while self.running:
            try:
                # Process messages with timeout
                try:
                    message = await asyncio.wait_for(
                        self._message_queue.get(), timeout=0.1
                    )
                    await self.handle_message(message)
                except TimeoutError:
                    pass

                # Send periodic heartbeat
                await self.heartbeat()

            except Exception as e:
                # Log error but keep running
                print(f"[{self.agent_id}] Error: {e}")
                await asyncio.sleep(0.1)

    async def stop(self) -> None:
        """Stop the tracker agent."""
        self.running = False

    async def handle_message(self, message: AgentMessage) -> None:
        """Process incoming messages."""
        if message.topic == "sensor":
            if message.message_type == "detection":
                await self._process_detections(message.payload)
        elif message.topic == "tracker":
            if message.message_type == "query":
                await self._handle_query(message.payload)
            elif message.message_type == "clear":
                self.tracks.clear()

    async def _process_detections(
        self,
        payload: dict,
    ) -> None:
        """
        Process new detections and update tracks.

        Parameters
        ----------
        payload : dict
            Detection data with keys:
            - timestamp: float
            - detections: list of [px, py, pz]
        """
        timestamp = payload.get("timestamp", time.time())
        detections_raw = payload.get("detections", [])

        if not detections_raw:
            return

        detections = [np.array(d, dtype=np.float64) for d in detections_raw]
        dt = timestamp - self._last_update_time if self._last_update_time > 0 else 0.1
        self._last_update_time = timestamp

        # Predict existing tracks
        confirmed_tracks = {
            tid: track
            for tid, track in self.tracks.items()
            if track.is_confirmed() and not track.is_terminated()
        }

        for track in confirmed_tracks.values():
            track.predict(dt)

        # Build cost matrix for data association
        if confirmed_tracks:
            track_ids = list(confirmed_tracks.keys())
            predicted_states = [
                confirmed_tracks[tid].ekf.x for tid in track_ids
            ]
            predicted_covs = [
                confirmed_tracks[tid].ekf.P for tid in track_ids
            ]

            cost_matrix = compute_cost_matrix(
                predicted_states,
                predicted_covs,
                detections,
                self._R,
            )

            # Hungarian assignment with gating
            matched_tracks, matched_dets, unmatched_tracks, unmatched_dets = (
                hungarian_assignment(
                    cost_matrix, gate_threshold=GATING_THRESHOLD
                )
            )

            # Update matched tracks
            for track_idx, det_idx in zip(matched_tracks, matched_dets, strict=False):
                tid = track_ids[track_idx]
                track = confirmed_tracks[tid]
                track.update(detections[det_idx], timestamp)

                # Publish track update
                await self.send_message(
                    topic="track",
                    message_type="update",
                    payload={
                        "track_id": tid,
                        "track": track.to_dict(),
                    },
                )

            # Handle unmatched tracks (misses)
            for track_idx in unmatched_tracks:
                tid = track_ids[track_idx]
                track = confirmed_tracks[tid]
                track.mark_miss()

                if track.is_terminated():
                    del self.tracks[tid]
                    await self.send_message(
                        topic="track",
                        message_type="lost",
                        payload={"track_id": tid},
                    )

            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                self._create_track(detections[det_idx], timestamp)

        else:
            # No existing tracks - create new tracks for all detections
            for det in detections:
                self._create_track(det, timestamp)

        # Enforce max tracks limit (remove oldest unconfirmed)
        if len(self.tracks) > self.max_tracks:
            self._prune_tracks()

    def _create_track(self, measurement: Vec3, timestamp: float) -> None:
        """Create a new track from a detection."""
        track_id = self._generate_track_id()
        track = Track.initialize(track_id, measurement, timestamp)
        self.tracks[track_id] = track

        # Publish new track
        asyncio.create_task(
            self.send_message(
                topic="track",
                message_type="new",
                payload={"track_id": track_id, "track": track.to_dict()},
            )
        )

    def _prune_tracks(self) -> None:
        """Remove oldest unconfirmed tracks to stay under limit."""
        # Sort by age, remove oldest unconfirmed first
        unconfirmed = [
            (tid, track)
            for tid, track in self.tracks.items()
            if not track.is_confirmed()
        ]
        unconfirmed.sort(key=lambda x: x[1].age)

        # Remove up to 10% of tracks
        n_remove = min(len(unconfirmed), max(1, len(self.tracks) // 10))
        for tid, _ in unconfirmed[:n_remove]:
            del self.tracks[tid]

    async def _handle_query(self, payload: dict) -> None:
        """Handle track data query."""
        track_id = payload.get("track_id")
        if track_id and track_id in self.tracks:
            track = self.tracks[track_id]
            await self.send_message(
                topic="track",
                message_type="response",
                payload={"track_id": track_id, "track": track.to_dict()},
            )

    def get_all_tracks(self) -> list[dict]:
        """Get all confirmed tracks as list of dicts."""
        return [
            track.to_dict()
            for track in self.tracks.values()
            if track.is_confirmed()
        ]

    def get_state(self) -> dict:
        """Get serializable state."""
        base = super().get_state()
        base["tracks"] = {
            tid: track.to_dict() for tid, track in self.tracks.items()
        }
        base["track_counter"] = self._track_counter
        return base
