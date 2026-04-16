"""Tracker service - Tier-1 agent pool manager."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

try:
    import zmq
    import zmq.asyncio
except ImportError:
    zmq = None  # type: ignore

from core.agents import TrackerAgent

if TYPE_CHECKING:
    from core.agents.base_agent import AgentMessage


class TrackerService:
    """
    Tracker service managing a pool of tracker agents.

    Responsibilities:
    - Subscribe to detection stream from ingest service
    - Distribute detections to tracker agents
    - Publish track updates to swarm intelligence service
    """

    def __init__(
        self,
        ingest_host: str = "localhost",
        ingest_port: int = 5555,
        output_port: int = 5556,
        n_agents: int = 4,
    ) -> None:
        """
        Initialize the tracker service.

        Parameters
        ----------
        ingest_host : str
            Hostname of ingest service.
        ingest_port : int
            ZMQ subscriber port for ingest.
        output_port : int
            ZMQ publisher port for track output.
        n_agents : int
            Number of tracker agent instances.
        """
        self.ingest_host = ingest_host
        self.ingest_port = ingest_port
        self.output_port = output_port
        self.n_agents = n_agents

        self._ctx = zmq.asyncio.Context() if zmq else None
        self._sub_socket = None
        self._pub_socket = None
        self._agents: list[TrackerAgent] = []
        self._running = False

    async def start(self) -> None:
        """Start the tracker service."""
        print(f"[tracker] Starting service with {self.n_agents} agents")

        if self._ctx:
            # Subscribe to ingest
            self._sub_socket = self._ctx.socket(zmq.SUB)
            self._sub_socket.connect(
                f"tcp://{self.ingest_host}:{self.ingest_port}"
            )
            self._sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            print(
                f"[tracker] Subscribed to tcp://{self.ingest_host}:{self.ingest_port}"
            )

            # Publish track updates
            self._pub_socket = self._ctx.socket(zmq.PUB)
            self._pub_socket.bind(f"tcp://*:{self.output_port}")
            print(f"[tracker] ZMQ publisher bound to port {self.output_port}")

        # Create tracker agents
        for i in range(self.n_agents):
            agent = TrackerAgent(agent_id=f"tracker_agent_{i}")
            # Override publish method to use ZMQ
            agent._publish = self._publish_message  # type: ignore
            self._agents.append(agent)

        self._running = True

        # Main loop
        while self._running:
            try:
                # Receive detections
                if self._sub_socket and zmq:
                    if await self._sub_socket.poll(100):  # 100ms timeout
                        message = await self._sub_socket.recv_json()
                        await self._process_detections(message)

                # Heartbeat from agents
                for agent in self._agents:
                    await agent.heartbeat()

            except Exception as e:
                print(f"[tracker] Error: {e}")
                await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the tracker service."""
        self._running = False
        for agent in self._agents:
            await agent.stop()
        if self._pub_socket:
            self._pub_socket.close()
        if self._sub_socket:
            self._sub_socket.close()
        if self._ctx:
            self._ctx.term()

    async def _process_detections(self, payload: dict) -> None:
        """Process incoming detections."""
        detections = payload.get("detections", [])
        timestamp = payload.get("timestamp", 0)

        if not detections:
            return

        # Distribute to first agent (simple load balancing)
        agent = self._agents[0]

        # Create message
        from core.agents.base_agent import AgentMessage

        message = AgentMessage(
            topic="sensor",
            sender_id="ingest_service",
            message_type="detection",
            payload={"timestamp": timestamp, "detections": detections},
        )

        await agent.handle_message(message)

        # Publish track updates
        for track in agent.get_all_tracks():
            await self._publish_message(
                AgentMessage(
                    topic="track",
                    sender_id=agent.agent_id,
                    message_type="update",
                    payload={"track": track},
                )
            )

    async def _publish_message(self, message: AgentMessage) -> None:
        """Publish message to ZMQ bus."""
        if self._pub_socket:
            await self._pub_socket.send_json(
                {
                    "topic": message.topic,
                    "sender_id": message.sender_id,
                    "message_type": message.message_type,
                    "payload": message.payload,
                    "timestamp": message.timestamp,
                }
            )


def main() -> None:
    """Entry point for tracker service."""
    service = TrackerService(
        ingest_host=os.environ.get("INGEST_HOST", "localhost"),
        ingest_port=int(os.environ.get("ZMQ_TRACKER_PORT", 5555)),
        output_port=int(os.environ.get("ZMQ_SWARM_PORT", 5556)),
    )

    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        asyncio.run(service.stop())


if __name__ == "__main__":
    main()
