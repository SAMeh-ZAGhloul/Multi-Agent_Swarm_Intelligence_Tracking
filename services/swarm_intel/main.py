"""Swarm Intelligence service - Tier-2 agent."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

try:
    import zmq
    import zmq.asyncio
except ImportError:
    zmq = None  # type: ignore

from core.agents import SwarmAgent
from core.agents.base_agent import AgentMessage

if TYPE_CHECKING:
    pass


class SwarmIntelService:
    """
    Swarm Intelligence service.

    Responsibilities:
    - Subscribe to track updates from tracker service
    - Build swarm graphs and classify behavior
    - Publish threat assessments to coordinator
    """

    def __init__(
        self,
        tracker_host: str = "localhost",
        tracker_port: int = 5556,
        output_port: int = 5557,
    ) -> None:
        """
        Initialize the swarm intelligence service.

        Parameters
        ----------
        tracker_host : str
            Hostname of tracker service.
        tracker_port : int
            ZMQ subscriber port for tracker.
        output_port : int
            ZMQ publisher port for swarm output.
        """
        self.tracker_host = tracker_host
        self.tracker_port = tracker_port
        self.output_port = output_port

        self._ctx = zmq.asyncio.Context() if zmq else None
        self._sub_socket = None
        self._pub_socket = None
        self._agent: SwarmAgent | None = None
        self._running = False

    async def start(self) -> None:
        """Start the swarm intelligence service."""
        print("[swarm_intel] Starting service")

        if self._ctx:
            # Subscribe to tracker
            self._sub_socket = self._ctx.socket(zmq.SUB)
            self._sub_socket.connect(
                f"tcp://{self.tracker_host}:{self.tracker_port}"
            )
            self._sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            print(
                f"[swarm_intel] Subscribed to tcp://{self.tracker_host}:{self.tracker_port}"
            )

            # Publish swarm analysis
            self._pub_socket = self._ctx.socket(zmq.PUB)
            self._pub_socket.bind(f"tcp://*:{self.output_port}")
            print(f"[swarm_intel] ZMQ publisher bound to port {self.output_port}")

        # Create swarm agent
        self._agent = SwarmAgent()
        self._agent._publish = self._publish_message  # type: ignore

        self._running = True

        while self._running:
            try:
                if self._sub_socket and zmq and await self._sub_socket.poll(100):
                    message = await self._sub_socket.recv_json()
                    await self._process_track_update(message)

                # Periodic analysis
                if self._agent and self._agent.swarms:
                    await self._agent._analyze_all_swarms()

                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"[swarm_intel] Error: {e}")
                await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the swarm intelligence service."""
        self._running = False
        if self._agent:
            await self._agent.stop()
        if self._pub_socket:
            self._pub_socket.close()
        if self._sub_socket:
            self._sub_socket.close()
        if self._ctx:
            self._ctx.term()

    async def _process_track_update(self, payload: dict) -> None:
        """Process incoming track update."""
        if not self._agent:
            return

        topic = payload.get("topic", "")
        if topic == "track":
            message = AgentMessage(
                topic=payload.get("topic", ""),
                sender_id=payload.get("sender_id", ""),
                message_type=payload.get("message_type", ""),
                payload=payload.get("payload", {}),
                timestamp=payload.get("timestamp", 0),
            )
            await self._agent.handle_message(message)

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

    def get_swarm_summary(self) -> list[dict]:
        """Get summary of all tracked swarms."""
        if self._agent:
            return self._agent.get_swarm_summary()
        return []


def main() -> None:
    """Entry point for swarm intelligence service."""
    service = SwarmIntelService(
        tracker_host=os.environ.get("TRACKER_HOST", "localhost"),
        tracker_port=int(os.environ.get("ZMQ_SWARM_PORT", 5556)),
        output_port=int(os.environ.get("ZMQ_COORD_PORT", 5557)),
    )

    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        asyncio.run(service.stop())


if __name__ == "__main__":
    main()
