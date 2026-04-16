# INFRASTRUCTURE.md — Deployment Topology & Sizing
# AEGIS-AI Platform

---

## 1. Deployment Profiles

Three deployment profiles are defined. Start with **Profile A** for the competition demo.

| Profile | Use Case | Hardware | Drones | Swarms |
|---|---|---|---|---|
| **A: Laptop Demo** | Competition demonstration | 1× developer laptop | up to 200 | up to 10 |
| **B: Edge Server** | Field deployment simulation | 1× workstation (16-core) | up to 500 | up to 20 |
| **C: Distributed** | Production concept | 3-node cluster | 2000+ | 50+ |

---

## 2. Profile A — Laptop Demo (Primary)

### Hardware Minimum Spec
```
CPU:    4-core (8-thread) Intel/AMD, 2.5GHz+
RAM:    16 GB
GPU:    Optional (GNN runs on CPU, <20ms for 50-node graph)
Disk:   10 GB free (models + logs)
OS:     Ubuntu 22.04 / macOS 13+ / Windows 11 (WSL2)
```

### Process Layout (single machine, Docker Compose)
```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Host                              │
│                                                                 │
│  ┌──────────────┐  ┌──────────────────────────────────────┐    │
│  │  Redis 7     │  │  PostgreSQL 16                       │    │
│  │  Port 6379   │  │  Port 5432                           │    │
│  │  64MB maxmem │  │  aegis database                      │    │
│  └──────────────┘  └──────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ingest-service         (1 replica)  Port 5550          │   │
│  │  tracker-service        (2 replicas) Port 5555, 5556    │   │
│  │  swarm-intel-service    (1 replica)  Port 5560          │   │
│  │  coordinator-service    (1 replica)  Port 5565          │   │
│  │  gateway-service        (1 replica)  Port 8000          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  streamlit-ui           (1 replica)  Port 8501          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  simulator              (1 replica)  -- no port --      │   │
│  │  (pushes data to ingest via ZMQ, no HTTP)               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### docker-compose.yml (Profile A)
```yaml
version: "3.9"
services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    command: redis-server --maxmemory 64mb --maxmemory-policy allkeys-lru

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: aegis
      POSTGRES_USER: aegis
      POSTGRES_PASSWORD: aegis_dev
    ports: ["5432:5432"]
    volumes: ["postgres_data:/var/lib/postgresql/data"]

  simulator:
    build: { context: ., dockerfile: infra/docker/simulator.Dockerfile }
    environment:
      - ZMQ_INGEST_ADDR=tcp://ingest:5550
      - SIM_SCENARIO=${SIM_SCENARIO:-saturation_attack}
      - SIM_DRONE_COUNT=${SIM_DRONE_COUNT:-50}
      - SIM_SEED=42
    depends_on: [ingest]

  ingest:
    build: { context: ., dockerfile: infra/docker/services.Dockerfile }
    command: python -m services.ingest.main
    ports: ["5550:5550"]
    environment: [REDIS_URL, DATABASE_URL]
    depends_on: [redis]

  tracker:
    build: { context: ., dockerfile: infra/docker/services.Dockerfile }
    command: python -m services.tracker.main
    deploy: { replicas: 2 }
    environment: [REDIS_URL, ZMQ_TRACKER_PORT]
    depends_on: [ingest, redis]

  swarm_intel:
    build: { context: ., dockerfile: infra/docker/services.Dockerfile }
    command: python -m services.swarm_intel.main
    environment: [REDIS_URL, GNN_MODEL_PATH]
    depends_on: [tracker, redis]

  coordinator:
    build: { context: ., dockerfile: infra/docker/services.Dockerfile }
    command: python -m services.coordinator.main
    environment: [REDIS_URL, DATABASE_URL]
    depends_on: [swarm_intel, redis]

  gateway:
    build: { context: ., dockerfile: infra/docker/services.Dockerfile }
    command: python -m services.gateway.main
    ports: ["8000:8000"]
    environment: [REDIS_URL, DATABASE_URL]
    depends_on: [coordinator, redis]

  ui:
    build: { context: ., dockerfile: infra/docker/ui.Dockerfile }
    command: streamlit run ui/app.py --server.port=8501 --server.address=0.0.0.0
    ports: ["8501:8501"]
    environment:
      - GATEWAY_WS_URL=ws://gateway:8000/ws/live
      - GATEWAY_REST_URL=http://gateway:8000
    depends_on: [gateway]

volumes:
  postgres_data:
```

### Resource Budget (Profile A, 200 drones, 10 swarms)
```
Process              CPU (avg)   RAM      Notes
───────────────────────────────────────────────────────
redis                0.5 core    64 MB    stream buffers
postgres             0.2 core    128 MB   track history
simulator            0.5 core    256 MB   Reynolds physics
ingest               0.3 core    128 MB   CI fusion
tracker ×2           1.0 core    192 MB   EKF × 100 drones each
swarm_intel          1.0 core    512 MB   GNN model in memory
coordinator          0.3 core    256 MB   PPO model
gateway              0.3 core    128 MB   WebSocket fanout
streamlit            0.5 core    384 MB   Plotly rendering
───────────────────────────────────────────────────────
TOTAL                4.6 cores   2.0 GB   fits 8-core/16GB laptop
```

---

## 3. Profile B — Edge Server (500 drones)

### Hardware
```
CPU:    16-core workstation (e.g., AMD Ryzen 9 7900X or Intel i9-13900K)
RAM:    32 GB
GPU:    NVIDIA RTX 3060+ (optional, CUDA for GNN batch inference)
```

### Scaling Changes vs Profile A
```yaml
tracker:
  deploy: { replicas: 5 }          # 100 drones per worker

swarm_intel:
  deploy: { replicas: 3 }          # parallel swarm analysis
  environment:
    - TORCH_DEVICE=cuda            # GPU inference if available
    - GNN_BATCH_SIZE=8             # batch multiple swarms

coordinator:
  deploy: { replicas: 2 }

redis:
  command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
```

### Resource Budget (Profile B, 500 drones, 20 swarms)
```
Process              CPU (avg)   RAM      Notes
───────────────────────────────────────────────────────
redis                1.0 core    512 MB
postgres             0.5 core    512 MB
simulator            1.0 core    512 MB
ingest               0.5 core    256 MB
tracker ×5           5.0 cores   480 MB   100 drones/worker
swarm_intel ×3       3.0 cores   1.5 GB   GNN × 3 workers
coordinator ×2       0.5 core    512 MB
gateway              0.5 core    256 MB
streamlit            1.0 core    512 MB
───────────────────────────────────────────────────────
TOTAL               13.0 cores   5.0 GB   fits 16-core/32GB
```

---

## 4. Profile C — Distributed Cluster (Concept)

### 3-Node Kubernetes Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Node 1: Control Plane + Data Services                       │
│  ├── Redis Cluster (3 shards)                                │
│  ├── PostgreSQL (primary)                                    │
│  ├── Ingest Service (3 pods)                                 │
│  └── Gateway Service (2 pods) + Ingress                      │
│                                                              │
│  Node 2: Compute-Heavy (Tracking + Swarm Intel)              │
│  ├── Tracker Workers (10 pods, HPA: scale on queue depth)    │
│  └── Swarm Intel Workers (5 pods, GPU-enabled)               │
│                                                              │
│  Node 3: Coordinator + UI                                    │
│  ├── Coordinator (2 pods, active-passive HA)                 │
│  ├── Streamlit UI (2 pods)                                   │
│  └── PostgreSQL (replica)                                    │
└──────────────────────────────────────────────────────────────┘
```

### Auto-Scaling Policy (HPA)
```yaml
# Tracker workers scale based on Redis stream lag
- metric: aegis:observations stream consumer lag
  scale_up:   lag > 5 frames   → add 1 tracker pod
  scale_down: lag < 1 frame    → remove 1 tracker pod
  min_replicas: 2
  max_replicas: 20

# Swarm intel scales on active swarm count
- metric: custom/active_swarm_count
  scale_up:   swarms > replicas * 3
  scale_down: swarms < replicas * 2
  min_replicas: 1
  max_replicas: 10
```

---

## 5. Performance Sizing Model

### EKF Throughput
```
Single EKF cycle (predict + update): ~0.3ms (NumPy, no JIT)
                                       ~0.08ms (with Numba JIT)
At 10Hz update rate, per worker:
  Sustainable drones = 1000ms / (0.3ms × 2) ≈ 166 drones/worker (no JIT)
  Sustainable drones = 1000ms / (0.08ms × 2) ≈ 625 drones/worker (Numba)

Profile A (2 workers, no JIT): 166 × 2 = 332 drones max
Profile B (5 workers, Numba):  625 × 5 = 3125 drones theoretical max
```

### Hungarian Assignment
```
LAPJV complexity: O(N²) average
N=100:   ~3ms   (Python + NumPy)
N=200:   ~11ms  (approaching 10Hz budget of 100ms)
N=500:   ~65ms  (requires partitioned assignment across workers)

Partitioning strategy: divide airspace into spatial sectors,
assign each sector to one tracker worker. Cross-sector hand-off
when drone moves between sectors (handled by SwarmGrouper).
```

### GNN Inference
```
50-node swarm graph:
  CPU (PyTorch):  ~15ms per swarm
  GPU (RTX 3060): ~2ms per swarm

Batch inference (8 swarms simultaneously):
  CPU:  ~80ms (sequential bottleneck)
  GPU:  ~5ms  (parallelized)

Profile A target (10 swarms, CPU):  ~150ms batched — within 500ms budget
Profile B target (20 swarms, GPU):  ~40ms batched  — comfortable margin
```

---

## 6. Data Retention Policy

```
Redis Streams (hot data):
  aegis:observations   → retain 10 seconds (100 frames)
  aegis:tracks         → retain 30 seconds (300 frames)
  aegis:swarm_intel    → retain 60 seconds
  aegis:threat         → retain 300 seconds

PostgreSQL (warm data):
  track_history        → retain 24 hours (for replay)
  swarm_events         → retain 7 days
  scenario_logs        → retain 30 days

File system (cold data):
  scenario recordings  → retain 90 days
  model checkpoints    → retain indefinitely (DVC tracked)
```

---

## 7. Health Monitoring

```
Endpoint: GET /health on each service (port 8000 for gateway)

Service health check response:
{
  "service": "tracker",
  "status": "healthy",         // healthy | degraded | down
  "active_tracks": 47,
  "queue_lag_frames": 0,
  "last_heartbeat": 1714123456.789,
  "uptime_seconds": 3847
}

Streamlit UI: sidebar shows service health indicators
  ● green  = all services healthy
  ● yellow = one service degraded
  ● red    = critical service down
```

---

## 8. Makefile Targets

```makefile
install:      pip install -e ".[dev]" && pre-commit install
dev:          docker compose up --build
sim-only:     python -m core.simulation.swarm_sim --scenario saturation_attack
test:         pytest tests/ -v --cov=core --cov-report=term-missing
lint:         ruff check . && mypy core/ services/
train-gnn:    python scripts/train_gnn.py --epochs 100
build-models: python scripts/generate_training_data.py && make train-gnn
clean:        docker compose down -v && find . -name __pycache__ -exec rm -rf {} +
profile:      python scripts/benchmark.py --drones 200 --duration 60
```
