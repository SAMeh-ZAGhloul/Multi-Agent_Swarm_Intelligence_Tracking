"""Global constants for AEGIS-AI core algorithms."""

from enum import IntEnum


class BehaviorClass(IntEnum):
    """Behavior classification for drone swarms."""

    TRANSIT = 0  # cohesive movement toward target
    ATTACK = 1  # converging, high speed, low spread
    SCATTER = 2  # dispersing (saturation maneuver)
    ENCIRCLE = 3  # pincer / encirclement formation
    DECOY = 4  # disordered, high separation weight
    UNKNOWN = 5  # insufficient data


# EKF State Vector: [px, py, pz, vx, vy, vz, omega]
STATE_DIM = 7
MEASUREMENT_DIM = 3

# Process noise standard deviations
PROCESS_NOISE_POS = 0.1  # m/s^2
PROCESS_NOISE_VEL = 0.05  # m/s^2
PROCESS_NOISE_OMEGA = 0.01  # rad/s^2

# Measurement noise standard deviations
MEASUREMENT_NOISE_POS = 1.0  # meters

# Gating threshold (chi-squared, dof=3, p=0.99)
GATING_THRESHOLD = 11.34

# Track lifecycle thresholds
CONFIRMATION_THRESHOLD = 3  # detections needed to confirm track
DELETION_THRESHOLD = 5  # missed detections before track deletion

# Reynolds flocking weights (baseline)
REYNOLDS_WEIGHT_SEPARATION = 0.3
REYNOLDS_WEIGHT_ALIGNMENT = 0.8
REYNOLDS_WEIGHT_COHESION = 0.6

# Swarm graph parameters
NEIGHBORHOOD_RADIUS = 50.0  # meters
MIN_NEIGHBOR_DISTANCE = 2.0  # meters

# PPO Coordinator parameters
COORDINATOR_ACTION_DIM = 4  # [intercept, jam, ignore, escalate]
COORDINATOR_OBS_DIM = 10  # [threat_score, distance, velocity, ...]
