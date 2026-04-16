# BRD — Business & Functional Requirements
# AEGIS-AI: Multi-Agent Counter-UAS Platform

**Version:** 1.0  
**Date:** April 2025  
**Competition:** Anti-Drone — 6th ITC International Conference, Air Defense College Egypt  
**Status:** Approved for Development

---

## 1. Executive Summary

AEGIS-AI addresses a documented gap in modern air defense: the inability of existing C-UAS
systems to track, classify, and respond to *swarms* of hostile UAVs rather than individual
targets. The system introduces a hierarchical multi-agent AI architecture that mirrors the
swarm intelligence of the threat, enabling real-time behavioral analysis and coordinated
response at operational scale.

**Winning Hypothesis:** Defense advantage in the drone era goes to whoever classifies swarm
intent first. AEGIS-AI wins the computation race, not the arms race.

---

## 2. Stakeholders

| Stakeholder | Role | Interest |
|---|---|---|
| Competition Panel (Air Defense College) | Evaluator | Technical novelty, implementability, military relevance |
| Demonstration Audience | Observer | Visual clarity, real-time behavior, intuitive UI |
| Development Team | Builder | Clear specs, testable requirements, clean architecture |

---

## 3. Functional Requirements

### FR-1: Multi-Target Tracking
| ID | Requirement | Priority |
|---|---|---|
| FR-1.1 | System SHALL track up to 500 simultaneous drone targets in real time | Critical |
| FR-1.2 | System SHALL maintain individual track identity through formation changes and occlusions | Critical |
| FR-1.3 | System SHALL assign unique track IDs and maintain track lifecycle (tentative → confirmed → lost → deleted) | Critical |
| FR-1.4 | System SHALL report position, velocity, and uncertainty ellipse for each track | Critical |
| FR-1.5 | System SHALL handle track birth (new detections) and track death (targets leaving airspace or destroyed) | High |
| FR-1.6 | System SHALL resolve track splitting (one track → two) and merging (two tracks → one) | High |

### FR-2: Swarm Detection and Grouping
| ID | Requirement | Priority |
|---|---|---|
| FR-2.1 | System SHALL automatically cluster individual tracks into swarm groups using spatial-temporal proximity | Critical |
| FR-2.2 | System SHALL maintain separate swarm identities when multiple hostile swarms operate simultaneously | Critical |
| FR-2.3 | System SHALL regroup swarms dynamically as formation geometry changes | High |
| FR-2.4 | System SHALL compute swarm-level features: centroid, spread radius, velocity coherence, convex hull area | Critical |
| FR-2.5 | System SHALL detect swarm split and merge events and update groupings within 2 seconds | High |

### FR-3: Behavior Classification
| ID | Requirement | Priority |
|---|---|---|
| FR-3.1 | System SHALL classify each active swarm into one of 6 behavioral categories every 500ms | Critical |
| FR-3.2 | Behavioral categories: TRANSIT, ATTACK, SCATTER, ENCIRCLE, DECOY, UNKNOWN | Critical |
| FR-3.3 | System SHALL provide a confidence score (0–1) for each classification | Critical |
| FR-3.4 | System SHALL detect behavioral transitions and timestamp them | High |
| FR-3.5 | System SHALL invert the Reynolds flocking model to estimate behavioral weight vector [w_sep, w_align, w_coh] | High |
| FR-3.6 | System SHALL predict swarm centroid trajectory for next 30 seconds | High |

### FR-4: Threat Assessment
| ID | Requirement | Priority |
|---|---|---|
| FR-4.1 | System SHALL compute a composite threat score (0–100) for each swarm every second | Critical |
| FR-4.2 | Threat score SHALL incorporate: attack probability, distance to asset, swarm size, trajectory confidence | Critical |
| FR-4.3 | System SHALL rank swarms by threat priority and maintain a prioritized threat queue | Critical |
| FR-4.4 | System SHALL generate alerts when threat score crosses configurable thresholds (warn=60, critical=80) | High |

### FR-5: Multi-Sensor Fusion
| ID | Requirement | Priority |
|---|---|---|
| FR-5.1 | System SHALL fuse observations from RF sensors, acoustic arrays, and computer vision | High |
| FR-5.2 | System SHALL operate in degraded mode with any single sensor unavailable | Critical |
| FR-5.3 | System SHALL use Covariance Intersection to handle unknown inter-sensor correlations | High |
| FR-5.4 | In simulation mode, system SHALL generate synthetic sensor observations for all modalities | Critical |

### FR-6: Response Coordination
| ID | Requirement | Priority |
|---|---|---|
| FR-6.1 | System SHALL recommend countermeasure allocation (jammer, directed energy, kinetic) per threat | High |
| FR-6.2 | Coordinator SHALL respect countermeasure availability and recharge constraints | High |
| FR-6.3 | Coordinator SHALL avoid recommending EM countermeasures overlapping civilian flight corridors | High |
| FR-6.4 | System SHALL log all response recommendations with justification scores | Medium |

### FR-7: User Interface
| ID | Requirement | Priority |
|---|---|---|
| FR-7.1 | System SHALL provide a Streamlit web UI accessible at localhost:8501 | Critical |
| FR-7.2 | UI SHALL display a real-time radar/map canvas showing all drone tracks and swarm boundaries | Critical |
| FR-7.3 | UI SHALL display swarm topology graph (nodes=drones, edges=interactions, color=behavior) | Critical |
| FR-7.4 | UI SHALL display threat score panel with ranked swarm list updated every 500ms | Critical |
| FR-7.5 | UI SHALL provide a scenario launcher with at least 5 named demo scenarios | Critical |
| FR-7.6 | UI SHALL use a light color theme throughout | Critical |
| FR-7.7 | UI SHALL display Reynolds parameter sliders allowing manual behavior injection | Medium |
| FR-7.8 | UI SHALL provide historical replay of any recorded scenario | Medium |
| FR-7.9 | UI SHALL display EKF uncertainty ellipses for each tracked drone | Medium |

### FR-8: Simulation Engine
| ID | Requirement | Priority |
|---|---|---|
| FR-8.1 | Simulator SHALL generate physically plausible drone swarm trajectories using Reynolds model | Critical |
| FR-8.2 | Simulator SHALL support all 6 behavior classes as scenario inputs | Critical |
| FR-8.3 | Simulator SHALL inject configurable sensor noise (Gaussian) into simulated observations | Critical |
| FR-8.4 | Simulator SHALL support up to 500 drones across 20 concurrent swarms | Critical |
| FR-8.5 | All demo scenarios SHALL be reproducible via fixed random seed | Critical |

---

## 4. Non-Functional Requirements (NFRs)

### NFR-1: Performance
| ID | Requirement | Target | Rationale |
|---|---|---|---|
| NFR-1.1 | EKF predict+update cycle per drone | < 1ms | 500 drones × 1ms = 500ms budget at 10Hz |
| NFR-1.2 | Hungarian assignment for N=100 drones | < 5ms | Must complete within sensor frame period |
| NFR-1.3 | GNN inference for 50-node swarm graph | < 20ms | Per-swarm, parallelizable |
| NFR-1.4 | End-to-end latency (sensor input → threat score) | < 100ms | Operational response requirement |
| NFR-1.5 | UI refresh rate | 500ms | Smooth without overwhelming operators |
| NFR-1.6 | Simulator throughput | 500 drones @ 20Hz | Demo requirement |

### NFR-2: Scalability
| ID | Requirement |
|---|---|
| NFR-2.1 | Tracker service SHALL scale horizontally; each process handles a partition of the drone ID space |
| NFR-2.2 | Adding tracker worker nodes SHALL require no code changes (config only) |
| NFR-2.3 | System SHALL support 1–20 concurrent swarms without architectural changes |
| NFR-2.4 | GNN inference SHALL be batchable across multiple swarms simultaneously |
| NFR-2.5 | Redis pub/sub SHALL handle burst of 10,000 state messages/second |

### NFR-3: Reliability
| ID | Requirement |
|---|---|
| NFR-3.1 | Track continuity rate SHALL exceed 95% (tracks maintained through 5% missed detections) |
| NFR-3.2 | Agent crash SHALL not cause data loss; state recoverable from Redis within 2 seconds |
| NFR-3.3 | System SHALL degrade gracefully: losing one sensor modality reduces accuracy, not availability |
| NFR-3.4 | All agents SHALL publish health heartbeat every 1 second |

### NFR-4: Accuracy
| ID | Requirement |
|---|---|
| NFR-4.1 | Position RMSE per track | < 0.5m (simulation); < 5m (real sensors) |
| NFR-4.2 | Behavior classification accuracy on held-out test set | > 88% |
| NFR-4.3 | Swarm formation recognition accuracy | > 85% |
| NFR-4.4 | False alarm rate (non-threat classified as threat) | < 5% |
| NFR-4.5 | Track-to-truth association rate | > 97% |

### NFR-5: Security (Demo Scope)
| ID | Requirement |
|---|---|
| NFR-5.1 | No external network calls during demonstration |
| NFR-5.2 | All credentials via environment variables |
| NFR-5.3 | No PII or classified data in logs |

---

## 5. Demo Scenarios (Required for Competition)

| Scenario ID | Name | Drones | Swarms | Behavior Sequence | Duration |
|---|---|---|---|---|---|
| SCN-01 | Single Swarm Transit | 20 | 1 | TRANSIT → ATTACK | 60s |
| SCN-02 | Saturation Attack | 50 | 1 | TRANSIT → SCATTER → ATTACK | 90s |
| SCN-03 | Pincer Maneuver | 30 | 2 | TRANSIT → ENCIRCLE | 75s |
| SCN-04 | Decoy + Real Attack | 40 | 2 | DECOY (swarm-1) + ATTACK (swarm-2) | 90s |
| SCN-05 | Adaptive Swarm (learning) | 60 | 3 | Mixed, behavior changes when drones destroyed | 120s |

Each scenario MUST:
- Be launchable from the UI Scenarios page with one click
- Display real-time track maintenance statistics
- Show behavior classification transitions with timestamps
- Generate a post-scenario performance report

---

## 6. Acceptance Criteria

The system is considered demo-ready when ALL of the following pass:

```
[ ] SCN-01 through SCN-05 complete without crashes
[ ] Track continuity > 95% in SCN-02 (worst-case saturation)
[ ] Behavior classification transitions are visible in UI within 2s of simulator change
[ ] Swarm topology graph updates in real time during all scenarios
[ ] Threat score correctly ranks ATTACK swarm above DECOY swarm in SCN-04
[ ] UI remains responsive (< 1s interaction latency) during SCN-05 (60 drones)
[ ] All unit tests pass (pytest --tb=short)
[ ] No mypy errors on core/ directory
```

---

## 7. Out of Scope (v1.0)

- Real hardware sensor integration (RF/acoustic/camera hardware) — simulation only
- Live countermeasure actuation (recommendations only, no hardware control)
- Multi-site / networked deployment (single node is sufficient for demo)
- Authentication / multi-user access
- Classified data handling
