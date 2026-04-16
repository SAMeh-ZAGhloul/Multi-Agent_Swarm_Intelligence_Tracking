"""Tier-2 Swarm Intelligence Agent."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING

import numpy as np

from ..constants import BehaviorClass
from ..swarm import build_swarm_graph
from ..swarm.behavior import BehaviorClassifier, behavior_to_threat_score
from .base_agent import AgentMessage, BaseAgent

if TYPE_CHECKING:
    import numpy.typing as npt


class SwarmAgent(BaseAgent):
    """
    Tier-2 Agent: Swarm intelligence and behavior classification.

    Responsibilities:
    - Receive track updates from Tier-1 (Tracker Agent)
    - Build swarm graphs from track positions
    - Classify swarm behavior using GNN or heuristics
    - Compute threat scores for each swarm
    - Alert Tier-3 (Coordinator) of high-threat swarms

    Messages Received:
    - track/update: Track state updates
    - swarm/query: Request for swarm analysis

    Messages Sent:
    - swarm/analysis: Swarm behavior analysis
    - swarm/threat: Threat alert for high-priority swarms
    """

    def __init__(
        self,
        agent_id: str | None = None,
        behavior_model_path: str | None = None,
    ) -> None:
        """
        Initialize the swarm agent.

        Parameters
        ----------
        agent_id : str | None
            Agent identifier. Generated if None.
        behavior_model_path : str | None
            Path to trained GNN model. Uses heuristics if None.
        """
        super().__init__(
            agent_id=agent_id or f"swarm_{uuid.uuid4().hex[:8]}",
            agent_type="swarm",
        )
        self.behavior_model_path = behavior_model_path
        self.classifier = BehaviorClassifier()

        # Swarm state: swarm_id -> {track_ids, positions, velocities}
        self.swarms: dict[str, dict] = {}

        # Track to swarm mapping
        self._track_to_swarm: dict[str, str] = {}

    async def start(self) -> None:
        """Start the swarm agent main loop."""
        self.running = True
        while self.running:
            try:
                try:
                    message = await asyncio.wait_for(
                        self._message_queue.get(), timeout=0.1
                    )
                    await self.handle_message(message)
                except TimeoutError:
                    pass

                # Periodic swarm analysis
                if self.swarms:
                    await self._analyze_all_swarms()

                await self.heartbeat()

            except Exception as e:
                print(f"[{self.agent_id}] Error: {e}")
                await asyncio.sleep(0.1)

    async def stop(self) -> None:
        """Stop the swarm agent."""
        self.running = False

    async def handle_message(self, message: AgentMessage) -> None:
        """Process incoming messages."""
        if message.topic == "track":
            if message.message_type == "update":
                await self._process_track_update(message.payload)
            elif message.message_type == "lost":
                await self._process_track_loss(message.payload)
        elif message.topic == "swarm":
            if message.message_type == "query":
                await self._handle_query(message.payload)
            elif message.message_type == "assign":
                await self._assign_track_to_swarm(message.payload)

    async def _process_track_update(self, payload: dict) -> None:
        """Process track update message."""
        track_data = payload.get("track", {})
        track_id = payload.get("track_id") or track_data.get("track_id")

        if not track_id:
            return

        position = np.array(track_data.get("position", [0, 0, 0]))
        velocity = np.array(track_data.get("velocity", [0, 0, 0]))

        # Check if track belongs to existing swarm
        swarm_id = self._track_to_swarm.get(track_id)

        if swarm_id is None:
            # Try to associate with existing swarm or create new
            swarm_id = self._find_or_create_swarm(track_id, position)
            self._track_to_swarm[track_id] = swarm_id

        # Update swarm state
        if swarm_id in self.swarms:
            swarm = self.swarms[swarm_id]
            # Update or add track
            if track_id not in swarm["track_ids"]:
                swarm["track_ids"].append(track_id)
            swarm["positions"][track_id] = position
            swarm["velocities"][track_id] = velocity

    async def _process_track_loss(self, payload: dict) -> None:
        """Process track loss message."""
        track_id = payload.get("track_id")
        if track_id and track_id in self._track_to_swarm:
            swarm_id = self._track_to_swarm[track_id]
            del self._track_to_swarm[track_id]

            if swarm_id in self.swarms:
                swarm = self.swarms[swarm_id]
                if track_id in swarm["track_ids"]:
                    swarm["track_ids"].remove(track_id)
                if track_id in swarm["positions"]:
                    del swarm["positions"][track_id]
                if track_id in swarm["velocities"]:
                    del swarm["velocities"][track_id]

                # Remove empty swarm
                if not swarm["track_ids"]:
                    del self.swarms[swarm_id]

    def _find_or_create_swarm(
        self, track_id: str, position: npt.NDArray[np.float64]
    ) -> str:
        """Find existing swarm or create new one."""
        # Simple distance-based association
        NEARBY_THRESHOLD = 50.0  # meters

        for swarm_id, swarm in self.swarms.items():
            if not swarm["positions"]:
                continue
            existing_pos = list(swarm["positions"].values())[0]
            dist = np.linalg.norm(position - existing_pos)
            if dist < NEARBY_THRESHOLD:
                return swarm_id

        # Create new swarm
        swarm_id = f"swarm_{uuid.uuid4().hex[:6]}"
        self.swarms[swarm_id] = {
            "track_ids": [],
            "positions": {},
            "velocities": {},
        }
        return swarm_id

    async def _analyze_all_swarms(self) -> None:
        """Analyze behavior for all swarms."""
        for swarm_id, swarm in self.swarms.items():
            if len(swarm["track_ids"]) < 2:
                continue

            positions = np.array(list(swarm["positions"].values()))
            velocities = np.array(list(swarm["velocities"].values()))

            if len(positions) < 2:
                continue

            # Classify behavior
            behavior = self.classifier.classify(positions, velocities)
            threat_score = behavior_to_threat_score(behavior)

            # Build graph features
            graph = build_swarm_graph(list(positions), list(velocities))

            # Send analysis
            await self.send_message(
                topic="swarm",
                message_type="analysis",
                payload={
                    "swarm_id": swarm_id,
                    "n_drones": len(swarm["track_ids"]),
                    "behavior": behavior.name,
                    "behavior_class": int(behavior),
                    "threat_score": threat_score,
                    "confidence": self.classifier.confidence,
                    "graph_density": graph.n_edges / max(1, graph.n_nodes),
                },
            )

            # Alert for high-threat swarms
            if threat_score > 0.7:
                await self.send_message(
                    topic="swarm",
                    message_type="threat",
                    payload={
                        "swarm_id": swarm_id,
                        "threat_score": threat_score,
                        "behavior": behavior.name,
                        "n_drones": len(swarm["track_ids"]),
                        "priority": "HIGH" if threat_score > 0.85 else "MEDIUM",
                    },
                )

    async def _handle_query(self, payload: dict) -> None:
        """Handle swarm query."""
        swarm_id = payload.get("swarm_id")
        if swarm_id and swarm_id in self.swarms:
            swarm = self.swarms[swarm_id]
            positions = np.array(list(swarm["positions"].values()))
            velocities = np.array(list(swarm["velocities"].values()))

            if len(positions) >= 2:
                behavior = self.classifier.classify(positions, velocities)
                threat_score = behavior_to_threat_score(behavior)
            else:
                behavior = BehaviorClass.UNKNOWN
                threat_score = 0.3

            await self.send_message(
                topic="swarm",
                message_type="response",
                payload={
                    "swarm_id": swarm_id,
                    "n_drones": len(swarm["track_ids"]),
                    "behavior": behavior.name,
                    "threat_score": threat_score,
                    "tracks": swarm["track_ids"],
                },
            )

    async def _assign_track_to_swarm(self, payload: dict) -> None:
        """Manually assign a track to a swarm."""
        track_id = payload.get("track_id")
        swarm_id = payload.get("swarm_id")

        if track_id and swarm_id:
            self._track_to_swarm[track_id] = swarm_id
            if swarm_id not in self.swarms:
                self.swarms[swarm_id] = {
                    "track_ids": [],
                    "positions": {},
                    "velocities": {},
                }

    def get_swarm_summary(self) -> list[dict]:
        """Get summary of all swarms."""
        summaries = []
        for swarm_id, swarm in self.swarms.items():
            positions = np.array(list(swarm["positions"].values()))
            if len(positions) >= 2:
                centroid = np.mean(positions, axis=0).tolist()
                behavior = self.classifier.classify(
                    positions,
                    np.array(list(swarm["velocities"].values())),
                )
                threat_score = behavior_to_threat_score(behavior)
            else:
                centroid = [0, 0, 0]
                behavior = BehaviorClass.UNKNOWN
                threat_score = 0.3

            summaries.append(
                {
                    "swarm_id": swarm_id,
                    "n_drones": len(swarm["track_ids"]),
                    "centroid": centroid,
                    "behavior": behavior.name,
                    "threat_score": threat_score,
                }
            )
        return summaries
