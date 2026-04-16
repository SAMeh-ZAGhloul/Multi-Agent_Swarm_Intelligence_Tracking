"""Sensor data ingestion service."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

try:
    import zmq
    import zmq.asyncio
except ImportError:
    zmq = None  # type: ignore

from core.simulation import ScenarioRunner, ScenarioType

if TYPE_CHECKING:
    pass


class IngestService:
    """
    Sensor data ingestion service.

    Responsibilities:
    - Generate simulated sensor data (or ingest from hardware)
    - Publish detections to tracker service via ZMQ
    - Support multiple sensor types (RF, acoustic, vision)
    """

    def __init__(
        self,
        use_simulation: bool = True,
        zmq_port: int = 5555,
        scenario: ScenarioType = ScenarioType.SATURATION_ATTACK,
    ) -> None:
        """
        Initialize the ingest service.

        Parameters
        ----------
        use_simulation : bool
            Use simulated data if True.
        zmq_port : int
            ZMQ publisher port.
        scenario : ScenarioType
            Simulation scenario to run.
        """
        self.use_simulation = use_simulation
        self.zmq_port = zmq_port
        self.scenario = scenario

        self._ctx = zmq.asyncio.Context() if zmq else None
        self._socket = None
        self._runner: ScenarioRunner | None = None
        self._running = False

    async def start(self) -> None:
        """Start the ingestion service."""
        print(f"[ingest] Starting service (simulation={self.use_simulation})")

        if self._ctx:
            self._socket = self._ctx.socket(zmq.PUB)
            self._socket.bind(f"tcp://*:{self.zmq_port}")
            print(f"[ingest] ZMQ publisher bound to port {self.zmq_port}")

        if self.use_simulation:
            self._runner = ScenarioRunner()
            self._runner.run_scenario(self.scenario)
            print(f"[ingest] Loaded scenario: {self.scenario.name}")

        self._running = True

        while self._running:
            try:
                await self._publish_detections()
                await asyncio.sleep(0.1)  # 10 Hz update rate
            except Exception as e:
                print(f"[ingest] Error: {e}")
                await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the ingestion service."""
        self._running = False
        if self._socket:
            self._socket.close()
        if self._ctx:
            self._ctx.term()

    async def _publish_detections(self) -> None:
        """Generate and publish detections."""
        timestamp = time.time()

        if self.use_simulation and self._runner:
            # Step simulation
            self._runner.simulator.step(dt=0.1)

            # Get detections with simulated noise
            detections = self._runner.simulator.get_detections()

            # Get all drone states for rich detection data
            self._runner.simulator.get_all_drones()

            payload = {
                "timestamp": timestamp,
                "detections": detections,
                "sensor_type": "simulated",
                "n_targets": len(detections),
            }

            if self._socket:
                message = f"DETECTION:{timestamp}".encode()
                self._socket.send(message)
                # Send JSON payload

                self._socket.send_json(payload)

            print(f"[ingest] Published {len(detections)} detections")
        else:
            # Hardware ingest would go here
            payload = {
                "timestamp": timestamp,
                "detections": [],
                "sensor_type": "hardware",
                "n_targets": 0,
            }
            print("[ingest] No hardware sensors connected")


def main() -> None:
    """Entry point for ingest service."""
    import argparse

    parser = argparse.ArgumentParser(description="AEGIS-AI Ingest Service")
    parser.add_argument(
        "--scenario",
        type=str,
        default="saturation_attack",
        choices=[s.name.lower() for s in ScenarioType],
        help="Simulation scenario",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="ZMQ publisher port",
    )
    args = parser.parse_args()

    scenario = ScenarioType[args.scenario.upper()]
    service = IngestService(
        use_simulation=True,
        zmq_port=args.port,
        scenario=scenario,
    )

    try:
        asyncio.run(service.start())
    except KeyboardInterrupt:
        asyncio.run(service.stop())


if __name__ == "__main__":
    main()
