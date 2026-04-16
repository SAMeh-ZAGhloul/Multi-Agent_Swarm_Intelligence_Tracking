"""FastAPI WebSocket gateway for UI communication."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    FastAPI = None  # type: ignore
    uvicorn = None  # type: ignore

try:
    import zmq
    import zmq.asyncio
except ImportError:
    zmq = None  # type: ignore

if TYPE_CHECKING:
    from typing import Any

# Create FastAPI app
app = FastAPI(title="AEGIS-AI Gateway", version="1.0.0")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GatewayService:
    """
    WebSocket gateway connecting UI to backend services.

    Responsibilities:
    - Accept WebSocket connections from Streamlit UI
    - Subscribe to all service topics via ZMQ
    - Forward messages to connected UI clients
    - Handle UI commands to backend services
    """

    def __init__(
        self,
        tracker_port: int = 5556,
        swarm_port: int = 5557,
        coordinator_port: int = 5558,
    ) -> None:
        """
        Initialize the gateway service.

        Parameters
        ----------
        tracker_port : int
            ZMQ subscriber port for tracker.
        swarm_port : int
            ZMQ subscriber port for swarm intelligence.
        coordinator_port : int
            ZMQ subscriber port for coordinator.
        """
        self.tracker_port = tracker_port
        self.swarm_port = swarm_port
        self.coordinator_port = coordinator_port

        self._ctx = zmq.asyncio.Context() if zmq else None
        self._sockets: list = []
        self._connected_clients: list[WebSocket] = []
        self._running = False

    async def start(self) -> None:
        """Start the gateway service."""
        print("[gateway] Starting service")

        if self._ctx and zmq:
            # Subscribe to all services
            for port, name in [
                (self.tracker_port, "tracker"),
                (self.swarm_port, "swarm"),
                (self.coordinator_port, "coordinator"),
            ]:
                socket = self._ctx.socket(zmq.SUB)
                socket.connect(f"tcp://localhost:{port}")
                socket.setsockopt_string(zmq.SUBSCRIBE, "")
                self._sockets.append(socket)
                print(f"[gateway] Subscribed to {name} on port {port}")

        self._running = True

        # Start message forwarding
        asyncio.create_task(self._forward_messages())

    async def stop(self) -> None:
        """Stop the gateway service."""
        self._running = False
        for socket in self._sockets:
            socket.close()
        if self._ctx:
            self._ctx.term()

    async def _forward_messages(self) -> None:
        """Forward ZMQ messages to connected WebSocket clients."""
        if not self._sockets:
            return

        while self._running:
            try:
                for socket in self._sockets:
                    if await socket.poll(50):  # 50ms timeout
                        message = await socket.recv_json()
                        await self._broadcast(message)

                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"[gateway] Forward error: {e}")
                await asyncio.sleep(1)

    async def _broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast message to all connected clients."""
        if not self._connected_clients:
            return

        disconnected: list[WebSocket] = []
        for client in self._connected_clients:
            try:
                await client.send_json(message)
            except Exception:
                disconnected.append(client)

        # Clean up disconnected clients
        for client in disconnected:
            if client in self._connected_clients:
                self._connected_clients.remove(client)


# Global gateway instance
gateway: GatewayService | None = None


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize gateway on startup."""
    global gateway
    gateway = GatewayService(
        tracker_port=int(os.environ.get("ZMQ_SWARM_PORT", 5556)),
        swarm_port=int(os.environ.get("ZMQ_COORD_PORT", 5557)),
        coordinator_port=5558,
    )
    await gateway.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cleanup on shutdown."""
    if gateway:
        await gateway.stop()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle WebSocket connections from UI."""
    await websocket.accept()
    if gateway:
        gateway._connected_clients.append(websocket)
        print(f"[gateway] Client connected. Total: {len(gateway._connected_clients)}")

    try:
        while True:
            # Receive commands from UI
            data = await websocket.receive_json()
            await handle_ui_command(data, websocket)
    except WebSocketDisconnect:
        print("[gateway] Client disconnected")
    finally:
        if gateway and websocket in gateway._connected_clients:
            gateway._connected_clients.remove(websocket)


async def handle_ui_command(data: dict, websocket: WebSocket) -> None:
    """Handle commands from UI."""
    command = data.get("command", "")

    if command == "get_status":
        # Return current system status
        await websocket.send_json(
            {
                "type": "status_response",
                "data": {
                    "services": ["ingest", "tracker", "swarm_intel", "coordinator"],
                    "status": "running",
                },
            }
        )
    elif command == "scenario":
        # Launch a scenario
        scenario = data.get("scenario", "")
        print(f"[gateway] Launching scenario: {scenario}")
        await websocket.send_json(
            {"type": "ack", "command": "scenario", "scenario": scenario}
        )
    else:
        await websocket.send_json({"type": "error", "message": f"Unknown command: {command}"})


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "gateway"}


@app.get("/api/tracks")
async def get_tracks() -> list[dict]:
    """Get current tracks."""
    return []


@app.get("/api/swarms")
async def get_swarms() -> list[dict]:
    """Get current swarms."""
    return []


def main() -> None:
    """Entry point for gateway service."""
    if uvicorn and FastAPI:
        uvicorn.run(
            app,
            host=os.environ.get("GATEWAY_HOST", "0.0.0.0"),
            port=int(os.environ.get("GATEWAY_PORT", 8000)),
        )
    else:
        print("FastAPI or uvicorn not installed")


if __name__ == "__main__":
    main()
