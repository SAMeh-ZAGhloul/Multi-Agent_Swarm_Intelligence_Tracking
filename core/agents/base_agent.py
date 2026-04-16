"""Base class for all AEGIS-AI agents."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import msgpack


@dataclass
class AgentMessage:
    """
    Message format for inter-agent communication.

    Attributes
    ----------
    topic : str
        Message topic/category for routing.
    sender_id : str
        ID of the sending agent.
    message_type : str
        Type of message (e.g., 'track_update', 'threat_alert').
    payload : dict
        Message payload (must be serializable).
    timestamp : float
        Unix timestamp of message creation.
    """

    topic: str
    sender_id: str
    message_type: str
    payload: dict
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())

    def serialize(self) -> bytes:
        """Serialize message to bytes using MessagePack."""
        data = {
            "topic": self.topic,
            "sender_id": self.sender_id,
            "message_type": self.message_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }
        return msgpack.packb(data, use_bin_type=True)

    @classmethod
    def deserialize(cls, data: bytes) -> AgentMessage:
        """Deserialize message from bytes."""
        parsed = msgpack.unpackb(data, raw=False)
        return cls(
            topic=parsed["topic"],
            sender_id=parsed["sender_id"],
            message_type=parsed["message_type"],
            payload=parsed["payload"],
            timestamp=parsed["timestamp"],
        )

    def to_json(self) -> str:
        """Serialize message to JSON string."""
        return json.dumps(
            {
                "topic": self.topic,
                "sender_id": self.sender_id,
                "message_type": self.message_type,
                "payload": self.payload,
                "timestamp": self.timestamp,
            }
        )


class BaseAgent(ABC):
    """
    Abstract base class for all AEGIS-AI agents.

    Agents communicate via ZMQ pub/sub and must be serializable
    for crash recovery.

    Attributes
    ----------
    agent_id : str
        Unique identifier for this agent.
    agent_type : str
        Type of agent (e.g., 'tracker', 'swarm', 'coordinator').
    running : bool
        Whether the agent is currently running.
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
    ) -> None:
        """
        Initialize the base agent.

        Parameters
        ----------
        agent_id : str
            Unique agent identifier.
        agent_type : str
            Agent type name.
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.running = False
        self._message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()

    @abstractmethod
    async def start(self) -> None:
        """Start the agent's main loop."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the agent gracefully."""
        pass

    @abstractmethod
    async def handle_message(self, message: AgentMessage) -> None:
        """
        Process an incoming message.

        Parameters
        ----------
        message : AgentMessage
            The message to process.
        """
        pass

    async def send_message(
        self,
        topic: str,
        message_type: str,
        payload: dict,
    ) -> AgentMessage:
        """
        Send a message to other agents.

        Parameters
        ----------
        topic : str
            Message topic.
        message_type : str
            Type of message.
        payload : dict
            Message payload.

        Returns
        -------
        AgentMessage
            The sent message.
        """
        message = AgentMessage(
            topic=topic,
            sender_id=self.agent_id,
            message_type=message_type,
            payload=payload,
        )
        await self._publish(message)
        return message

    async def _publish(self, message: AgentMessage) -> None:
        """
        Publish a message to the ZMQ pub/sub bus.

        Subclasses must implement this method.
        """
        # Default implementation - override in subclasses
        pass

    async def _subscribe(self, topic: str) -> None:
        """
        Subscribe to a message topic.

        Subclasses must implement this method.
        """
        # Default implementation - override in subclasses
        pass

    def get_state(self) -> dict[str, Any]:
        """
        Get serializable state for crash recovery.

        Returns
        -------
        dict
            Agent state dictionary.
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "running": self.running,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """
        Restore state from crash recovery.

        Parameters
        ----------
        state : dict
            Previously saved state.
        """
        self.running = state.get("running", False)

    async def heartbeat(self) -> None:
        """Send heartbeat message to health monitoring."""
        await self.send_message(
            topic="health",
            message_type="heartbeat",
            payload={
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "status": "alive",
            },
        )
