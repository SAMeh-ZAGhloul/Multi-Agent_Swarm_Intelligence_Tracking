# API.md — Internal Service Contracts
# AEGIS-AI Platform

---

## 1. Gateway REST API (FastAPI, Port 8000)

Base URL: `http://localhost:8000`

### Scenario Control
```
POST /scenarios/start
Body:  { "scenario_id": "SCN-02", "drone_count": 50, "noise_level": 0.5, "seed": 42 }
Resp:  { "run_id": "uuid", "started_at": float, "scenario": {...} }

POST /scenarios/pause
POST /scenarios/resume
POST /scenarios/stop
GET  /scenarios/status        → { "running": bool, "run_id": str, "elapsed": float }
GET  /scenarios/list          → [{ "id": str, "name": str, "description": str }]
```

### Live State (polling alternative to WebSocket)
```
GET /state/snapshot           → UISnapshot (latest aggregated state)
GET /state/tracks             → List[TrackState] (all confirmed tracks)
GET /state/swarms             → List[SwarmIntelReport] (all active swarms)
GET /state/threats            → ThreatAssessment (latest)
```

### History & Analytics
```
GET /history/tracks?run_id=&start=&end=    → paginated TrackState list
GET /history/events?run_id=                → List[AlertEvent]
GET /analytics/performance?run_id=         → PerformanceReport
```

### Configuration
```
GET  /config                   → current SystemConfig
POST /config                   → update config (hot reload)
```

### Health
```
GET /health                    → ServiceHealth (gateway)
GET /health/all                → Dict[service_name, ServiceHealth]
```

---

## 2. WebSocket Live Feed (Gateway)

```
URL: ws://localhost:8000/ws/live

Connection: no auth required (demo mode)

Server → Client messages (JSON, every 500ms):
{
  "type": "snapshot",
  "timestamp": float,
  "tracks": [TrackState, ...],         // all confirmed tracks
  "swarms": [SwarmIntelReport, ...],   // all active swarm reports
  "threat": ThreatAssessment,
  "alerts": [AlertEvent, ...],         // new since last snapshot
  "stats": {
    "total_drones": int,
    "total_swarms": int,
    "tracker_lag_ms": float,
    "gnn_inference_ms": float,
  }
}

Client → Server messages:
{ "type": "ping" }
{ "type": "subscribe", "topics": ["tracks", "swarms", "threats", "alerts"] }
{ "type": "scenario_cmd", "action": "start"|"pause"|"stop", "params": {...} }
```

---

## 3. ZMQ Internal Bus

All services use ZeroMQ for inter-process communication.
Message format: MessagePack serialization.

```
Simulator → Ingest:
  Socket: PUSH/PULL
  Address: tcp://ingest:5550
  Message: ObservationMessage

Ingest → Tracker workers:
  Via Redis Stream: aegis:observations
  Consumer group: "tracker-pool"
  Each worker claims a partition of drone IDs

Tracker workers → Redis:
  Stream: aegis:tracks  (write)
  Key format: track_id hashed to worker

Redis → SwarmIntel:
  Stream: aegis:tracks  (read, consumer group: "swarm-intel")
  Stream: aegis:swarm_groups (read)

SwarmIntel → Redis:
  Stream: aegis:swarm_intel  (write)

Redis → Coordinator:
  Stream: aegis:swarm_intel  (read)

Coordinator → Redis:
  Stream: aegis:threat  (write)
  Key:    aegis:alerts   (List, LPUSH)

Redis → Gateway:
  Subscribes to all streams + pubsub channels
```

---

## 4. Pydantic Model Reference

```python
# core/models.py — the single source of truth for all message types

from pydantic import BaseModel
from enum import IntEnum
import numpy as np

class BehaviorClass(IntEnum):
    TRANSIT   = 0
    ATTACK    = 1
    SCATTER   = 2
    ENCIRCLE  = 3
    DECOY     = 4
    UNKNOWN   = 5

class TrackStatus(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING  = "coasting"
    DELETED   = "deleted"

class AlertLevel(str, Enum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"

class CountermeasureType(str, Enum):
    RF_JAMMER      = "rf_jammer"
    DIRECTED_ENERGY = "directed_energy"
    KINETIC        = "kinetic"
    NONE           = "none"

class Detection(BaseModel):
    det_id:     str
    px: float; py: float; pz: float
    sigma_x: float; sigma_y: float; sigma_z: float
    rf_dbm:     float = -999.0
    confidence: float = 1.0

class ObservationMessage(BaseModel):
    obs_id:     str
    sensor:     str
    timestamp:  float
    detections: list[Detection]

class TrackState(BaseModel):
    track_id:   str
    swarm_id:   str | None
    timestamp:  float
    state:      list[float]      # len=7: [px,py,pz,vx,vy,vz,omega]
    covariance: list[float]      # len=49: P matrix flattened
    status:     TrackStatus
    hits:       int
    misses:     int
    behavior:   BehaviorClass = BehaviorClass.UNKNOWN

class SwarmIntelReport(BaseModel):
    swarm_id:            str
    timestamp:           float
    drone_count:         int
    behavior_class:      BehaviorClass
    behavior_confidence: float
    reynolds_weights:    list[float]   # len=3: [w_sep, w_align, w_coh]
    threat_score:        float
    centroid:            list[float]   # len=3
    spread_radius:       float
    velocity_coherence:  float
    approach_rate:       float
    trajectory_30s:      list[list[float]]  # 30 × 3
    graph_edges:         list[tuple[str,str]]
    alert_level:         AlertLevel = AlertLevel.INFO

class ThreatRecord(BaseModel):
    swarm_id:        str
    threat_score:    float
    rank:            int
    countermeasure:  CountermeasureType
    alert_level:     AlertLevel

class AlertEvent(BaseModel):
    level:      AlertLevel
    swarm_id:   str
    message:    str
    timestamp:  float

class ThreatAssessment(BaseModel):
    timestamp:       float
    ranked_threats:  list[ThreatRecord]
    alerts:          list[AlertEvent]

class UISnapshot(BaseModel):
    timestamp:  float
    tracks:     list[TrackState]
    swarms:     list[SwarmIntelReport]
    threat:     ThreatAssessment | None
    stats:      dict

class ServiceHealth(BaseModel):
    service:          str
    status:           str        # "healthy" | "degraded" | "down"
    active_tracks:    int = 0
    queue_lag_frames: int = 0
    last_heartbeat:   float
    uptime_seconds:   float

class PerformanceReport(BaseModel):
    run_id:                   str
    scenario_id:              str
    duration_seconds:         float
    total_drones:             int
    track_continuity_rate:    float
    behavior_accuracy:        float
    mean_latency_ms:          float
    p95_latency_ms:           float
    false_alarm_rate:         float
    alerts_generated:         int
```

---

## 5. Streamlit ↔ Gateway Communication Pattern

```python
# ui/components/data_client.py

import asyncio, json, websockets
import streamlit as st

def get_snapshot() -> UISnapshot | None:
    """
    Fetch the latest snapshot.
    Uses st.session_state to cache the WebSocket connection.
    Falls back to REST polling if WebSocket is unavailable.
    """
    if "last_snapshot" not in st.session_state:
        st.session_state.last_snapshot = None

    try:
        resp = httpx.get(f"{GATEWAY_URL}/state/snapshot", timeout=0.4)
        st.session_state.last_snapshot = UISnapshot(**resp.json())
    except Exception:
        pass  # return last known good snapshot

    return st.session_state.last_snapshot
```

---

## 6. Simulation Data Generation Contract

```python
# core/simulation/scenarios.py

SCENARIOS: dict[str, ScenarioConfig] = {
    "SCN-01": ScenarioConfig(
        name="Single Swarm Transit",
        drone_count=20, swarm_count=1,
        behavior_sequence=[
            BehaviorPhase(behavior=BehaviorClass.TRANSIT, duration_s=30),
            BehaviorPhase(behavior=BehaviorClass.ATTACK,  duration_s=30),
        ],
        noise_sigma=1.0, seed=1001,
    ),
    "SCN-02": ScenarioConfig(
        name="Saturation Attack",
        drone_count=50, swarm_count=1,
        behavior_sequence=[
            BehaviorPhase(behavior=BehaviorClass.TRANSIT, duration_s=30),
            BehaviorPhase(behavior=BehaviorClass.SCATTER, duration_s=20),
            BehaviorPhase(behavior=BehaviorClass.ATTACK,  duration_s=40),
        ],
        noise_sigma=1.5, seed=1002,
    ),
    # SCN-03, SCN-04, SCN-05 defined similarly
}
```
