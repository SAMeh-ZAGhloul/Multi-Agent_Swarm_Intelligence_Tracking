"""Tier-3 Response Coordinator Agent."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from .base_agent import AgentMessage, BaseAgent

if TYPE_CHECKING:
    pass


class ResponseAction(IntEnum):
    """Available response actions."""

    INTERCEPT = 0  # Deploy interceptor drone
    JAM = 1  # Activate RF jamming
    IGNORE = 2  # Monitor only (low threat)
    ESCALATE = 3  # Alert human operator


@dataclass
class ThreatAssessment:
    """Computed threat assessment."""

    swarm_id: str
    threat_score: float
    recommended_action: ResponseAction
    confidence: float
    n_drones: int
    behavior: str


class CoordinatorAgent(BaseAgent):
    """
    Tier-3 Agent: Response coordination and decision making.

    Responsibilities:
    - Receive threat alerts from Tier-2 (Swarm Agent)
    - Compute optimal response using PPO or rule-based policy
    - Coordinate response actions (intercept, jam, escalate)
    - Log all decisions for audit trail

    Messages Received:
    - swarm/threat: High-threat swarm alert
    - coordinator/command: Manual override command

    Messages Sent:
    - response/action: Coordinated response action
    - response/status: Response execution status
    """

    def __init__(
        self,
        agent_id: str | None = None,
        policy_model_path: str | None = None,
        auto_respond: bool = True,
    ) -> None:
        """
        Initialize the coordinator agent.

        Parameters
        ----------
        agent_id : str | None
            Agent identifier. Generated if None.
        policy_model_path : str | None
            Path to trained PPO model. Uses rules if None.
        auto_respond : bool
            Whether to automatically respond to threats.
        """
        super().__init__(
            agent_id=agent_id or f"coordinator_{uuid.uuid4().hex[:8]}",
            agent_type="coordinator",
        )
        self.policy_model_path = policy_model_path
        self.auto_respond = auto_respond

        # Active threats and responses
        self.active_threats: dict[str, ThreatAssessment] = {}
        self.active_responses: dict[str, ResponseAction] = {}

        # Response history for audit
        self._decision_log: list[dict] = []

    async def start(self) -> None:
        """Start the coordinator main loop."""
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

                # Process active threats
                if self.active_threats and self.auto_respond:
                    await self._process_active_threats()

                await self.heartbeat()

            except Exception as e:
                print(f"[{self.agent_id}] Error: {e}")
                await asyncio.sleep(0.1)

    async def stop(self) -> None:
        """Stop the coordinator agent."""
        self.running = False

    async def handle_message(self, message: AgentMessage) -> None:
        """Process incoming messages."""
        if message.topic == "swarm":
            if message.message_type == "threat":
                await self._process_threat_alert(message.payload)
            elif message.message_type == "analysis":
                await self._process_swarm_analysis(message.payload)
        elif message.topic == "coordinator":
            if message.message_type == "command":
                await self._handle_manual_command(message.payload)
            elif message.message_type == "query":
                await self._handle_query(message.payload)

    async def _process_threat_alert(self, payload: dict) -> None:
        """Process high-threat swarm alert."""
        swarm_id = payload.get("swarm_id")
        if not swarm_id:
            return

        threat_score = payload.get("threat_score", 0.0)
        behavior = payload.get("behavior", "UNKNOWN")
        n_drones = payload.get("n_drones", 0)
        priority = payload.get("priority", "MEDIUM")

        # Determine recommended action
        action = self._determine_action(
            threat_score=threat_score,
            behavior=behavior,
            n_drones=n_drones,
            priority=priority,
        )

        assessment = ThreatAssessment(
            swarm_id=swarm_id,
            threat_score=threat_score,
            recommended_action=action,
            confidence=payload.get("confidence", 0.5),
            n_drones=n_drones,
            behavior=behavior,
        )

        self.active_threats[swarm_id] = assessment

        # Log decision
        self._decision_log.append(
            {
                "timestamp": asyncio.get_event_loop().time(),
                "swarm_id": swarm_id,
                "threat_score": threat_score,
                "action": action.name,
            }
        )

        # Execute response if auto-respond enabled
        if self.auto_respond:
            await self._execute_response(assessment)

    async def _process_swarm_analysis(self, payload: dict) -> None:
        """Process routine swarm analysis."""
        threat_score = payload.get("threat_score", 0.0)
        swarm_id = payload.get("swarm_id")

        # Only track if threat score is elevated
        if threat_score > 0.5 and swarm_id:
            behavior = payload.get("behavior", "UNKNOWN")
            n_drones = payload.get("n_drones", 0)

            action = self._determine_action(
                threat_score=threat_score,
                behavior=behavior,
                n_drones=n_drones,
                priority="LOW",
            )

            self.active_threats[swarm_id] = ThreatAssessment(
                swarm_id=swarm_id,
                threat_score=threat_score,
                recommended_action=action,
                confidence=payload.get("confidence", 0.5),
                n_drones=n_drones,
                behavior=behavior,
            )

    def _determine_action(
        self,
        threat_score: float,
        behavior: str,
        n_drones: int,
        priority: str,
    ) -> ResponseAction:
        """
        Determine response action based on threat assessment.

        Rule-based policy (replaced by PPO in production).
        """
        # ESCALATE: Very high threat or attack behavior
        if threat_score > 0.9 or behavior == "ATTACK":
            return ResponseAction.ESCALATE

        # INTERCEPT: High threat with coherent swarm
        if threat_score > 0.7 and behavior in ("ATTACK", "ENCIRCLE"):
            return ResponseAction.INTERCEPT

        # JAM: Scattering or decoy swarms
        if threat_score > 0.6 and behavior in ("SCATTER", "DECOY"):
            return ResponseAction.JAM

        # IGNORE: Low threat
        if threat_score < 0.4:
            return ResponseAction.IGNORE

        # Default: monitor and escalate if needed
        return ResponseAction.ESCALATE

    async def _execute_response(self, assessment: ThreatAssessment) -> None:
        """Execute the recommended response action."""
        action = assessment.recommended_action

        # Send response command
        await self.send_message(
            topic="response",
            message_type="action",
            payload={
                "swarm_id": assessment.swarm_id,
                "action": action.name,
                "action_id": int(action),
                "threat_score": assessment.threat_score,
                "confidence": assessment.confidence,
                "n_drones": assessment.n_drones,
            },
        )

        self.active_responses[assessment.swarm_id] = action

        # Update status
        await self.send_message(
            topic="response",
            message_type="status",
            payload={
                "swarm_id": assessment.swarm_id,
                "action": action.name,
                "status": "EXECUTED",
            },
        )

    async def _process_active_threats(self) -> None:
        """Re-evaluate and update active threats."""
        # Decay threat scores over time
        to_remove = []
        for swarm_id, assessment in self.active_threats.items():
            # Reduce threat score over time if no updates
            assessment.threat_score *= 0.95
            if assessment.threat_score < 0.2:
                to_remove.append(swarm_id)

        for swarm_id in to_remove:
            del self.active_threats[swarm_id]
            # Send de-escalation message
            await self.send_message(
                topic="response",
                message_type="status",
                payload={
                    "swarm_id": swarm_id,
                    "status": "DE_ESCALATED",
                },
            )

    async def _handle_manual_command(self, payload: dict) -> None:
        """Handle manual override command."""
        action = payload.get("action")
        swarm_id = payload.get("swarm_id")

        if action and swarm_id:
            try:
                response_action = ResponseAction[action.upper()]
                await self.send_message(
                    topic="response",
                    message_type="action",
                    payload={
                        "swarm_id": swarm_id,
                        "action": response_action.name,
                        "manual_override": True,
                    },
                )
                self.active_responses[swarm_id] = response_action
            except KeyError:
                pass

    async def _handle_query(self, payload: dict) -> None:
        """Handle status query."""
        query_type = payload.get("type", "summary")

        if query_type == "summary":
            await self.send_message(
                topic="coordinator",
                message_type="response",
                payload={
                    "active_threats": len(self.active_threats),
                    "active_responses": len(self.active_responses),
                    "threats": [
                        {
                            "swarm_id": a.swarm_id,
                            "threat_score": a.threat_score,
                            "action": a.recommended_action.name,
                        }
                        for a in self.active_threats.values()
                    ],
                },
            )
        elif query_type == "history":
            limit = payload.get("limit", 10)
            await self.send_message(
                topic="coordinator",
                message_type="response",
                payload={"history": self._decision_log[-limit:]},
            )

    def get_status(self) -> dict:
        """Get coordinator status."""
        return {
            "active_threats": len(self.active_threats),
            "active_responses": len(self.active_responses),
            "auto_respond": self.auto_respond,
        }
