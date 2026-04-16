"""Response Coordinator service - Tier-3 agent."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

try:
    import zmq
    import zmq.asyncio
except ImportError:
    zmq = None  # type: ignore

from core.agents import CoordinatorAgent
from core.agents.base_agent import AgentMessage

if TYPE_CHECKING:
    pass


class CoordinatorService:
    """
    Response Coordinator service.

    Responsibilities:
    - Receive threat alerts from swarm intelligence
    - Determine optimal response action
    - Coordinate response execution
    - Provide API for human operator override
    """

    def __init__(
        self,
        swarm_host: str = "localhost",
        swarm_port: int = 5557,
        output_port: int = 5558,
        auto_respond: bool = True,
    ) -> None:
        """
        Initialize the coordinator service.

        Parameters
        ----------
        swarm_host : str
            Hostname of swarm intelligence service.
        swarm_port : int
            ZMQ subscriber port for swarm.
        output_port : int
            ZMQ publisher port for response output.
        auto_respond : bool
            Whether to automatically respond to threats.
        """
        self.swarm_host = swarm_host
        self.swarm_port = swarm_port
        self.output_port = output_port
        self.auto_respond = auto_respond

        self._ctx = zmq.asyncio.Context() if zmq else None
        self._sub_socket = None
        self._pub_socket = None
        self._agent: CoordinatorAgent | None = None
        self._running = False

    async def start(self) -> None:
        """Start the coordinator service."""
        print(f"[coordinator] Starting service (auto_respond={self.auto_respond})")

        if self._ctx:
            # Subscribe to swarm intelligence
            self._sub_socket = self._ctx.socket(zmq.SUB)
            self._sub_socket.connect(
                f"tcp://{self.swarm_host}:{self.swarm_port}"
            )
            self._sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            print(
                f"[coordinator] Subscribed to tcp://{self.swarm_host}:{self.swarm_port}"
            )

            # Publish response actions
            self._pub_socket = self._ctx.socket(zmq.PUB)
            self._pub_socket.bind(f"tcp://*:{self.output_port}")
            print(f"[coordinator] ZMQ publisher bound to port {self.output_port}")

        # Create coordinator agent
        self._agent = CoordinatorAgent(auto_respond=self.auto_respond)
        self._agent._publish = self._publish_message  # type: ignore

        self._running = True

        while self._running:
            try:
                if self._sub_socket and zmq and await self._sub_socket.poll(100):
                    message = await self._sub_socket.recv_json()
                    await self._process_threat(message)

                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"[coordinator] Error: {e}")
                await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the coordinator service."""
        self._running = False
        if self._agent:
            await self._agent.stop()
        if self._pub_socket:
            self._pub_socket.close()
        if self._sub_socket:
            self._sub_socket.close()
        if self._ctx:
            self._ctx.term()

    async def _process_threat(self, payload: dict) -> None:
        """Process incoming threat alert."""
        if not self._agent:
            return

        topic = payload.get("topic", "")
        if topic == "swarm" and payload.get("message_type") == "threat":
            message = AgentMessage(
                topic="swarm",
                sender_id=payload.get("sender_id", ""),
                message_type="threat",
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

    def get_status(self) -> dict:
        """Get coordinator status."""
        if self._agent:
            return self._agent.get_status()
        return {"active_threats": 0, "active_responses": 0}


def main() -> None:
    """Entry point for coordinator service."""
    service = CoordinatorService(
        swarm_host=os.environ.get("SWARM_HOST", "localhost"),
        swarm_port=int(os.environ.get("ZMQ_COORD_PORT", 5557)),
        auto_respond=os.environ.get("AUTO_RESPOND", "true").lower() == "true",
    )

    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        asyncio.run(service.stop())


if __name__ == "__main__":
    main()
