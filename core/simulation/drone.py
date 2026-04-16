"""Simulated drone with Reynolds flocking physics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..constants import BehaviorClass

Vec3 = npt.NDArray[np.float64]


@dataclass
class DroneState:
    """
    Complete state of a simulated drone.

    Attributes
    ----------
    drone_id : str
        Unique identifier (format: "swarm_{id}_drone_{n}").
    swarm_id : str
        Parent swarm identifier.
    timestamp : float
        Unix timestamp of this state.
    px, py, pz : float
        Position in meters (ENU frame).
    vx, vy, vz : float
        Velocity in m/s.
    behavior : BehaviorClass
        Current behavior classification.
    rf_power_dbm : float
        Simulated RF signal strength.
    is_alive : bool
        Whether drone is operational.
    """

    drone_id: str
    swarm_id: str
    timestamp: float
    px: float
    py: float
    pz: float
    vx: float
    vy: float
    vz: float
    behavior: BehaviorClass = BehaviorClass.TRANSIT
    rf_power_dbm: float = -50.0
    is_alive: bool = True

    @property
    def position(self) -> Vec3:
        """Get position as numpy array."""
        return np.array([self.px, self.py, self.pz], dtype=np.float64)

    @position.setter
    def position(self, value: Vec3) -> None:
        """Set position from numpy array."""
        self.px, self.py, self.pz = float(value[0]), float(value[1]), float(value[2])

    @property
    def velocity(self) -> Vec3:
        """Get velocity as numpy array."""
        return np.array([self.vx, self.vy, self.vz], dtype=np.float64)

    @velocity.setter
    def velocity(self, value: Vec3) -> None:
        """Set velocity from numpy array."""
        self.vx, self.vy, self.vz = float(value[0]), float(value[1]), float(value[2])

    @property
    def speed(self) -> float:
        """Get current speed (magnitude of velocity)."""
        return float(np.linalg.norm(self.velocity))

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "drone_id": self.drone_id,
            "swarm_id": self.swarm_id,
            "timestamp": self.timestamp,
            "position": [self.px, self.py, self.pz],
            "velocity": [self.vx, self.vy, self.vz],
            "behavior": self.behavior.name,
            "rf_power_dbm": self.rf_power_dbm,
            "is_alive": self.is_alive,
        }


class Drone:
    """
    Simulated drone agent with Reynolds flocking physics.

    Attributes
    ----------
    state : DroneState
        Current drone state.
    target_velocity : Vec3
        Desired velocity from flocking forces.
    """

    def __init__(
        self,
        drone_id: str,
        swarm_id: str,
        initial_position: Vec3 | None = None,
        initial_velocity: Vec3 | None = None,
        behavior: BehaviorClass = BehaviorClass.TRANSIT,
    ) -> None:
        """
        Initialize a simulated drone.

        Parameters
        ----------
        drone_id : str
            Unique drone identifier.
        swarm_id : str
            Parent swarm identifier.
        initial_position : Vec3 | None
            Starting position. Random if None.
        initial_velocity : Vec3 | None
            Starting velocity. Zero if None.
        behavior : BehaviorClass
            Initial behavior class.
        """
        self.state = DroneState(
            drone_id=drone_id,
            swarm_id=swarm_id,
            timestamp=0.0,
            px=0.0,
            py=0.0,
            pz=0.0,
            vx=0.0,
            vy=0.0,
            vz=0.0,
            behavior=behavior,
        )

        if initial_position is not None:
            self.state.position = initial_position
        else:
            # Random position in 100x100x50 volume
            self.state.position = np.random.uniform(
                -50, 50, 3
            ).astype(np.float64)

        if initial_velocity is not None:
            self.state.velocity = initial_velocity

        self.target_velocity = self.state.velocity.copy()

        # Behavior-specific parameters
        self._behavior_params = self._get_behavior_params(behavior)

    def _get_behavior_params(self, behavior: BehaviorClass) -> dict:
        """Get behavior-specific flocking parameters."""
        # Reynolds weights: [separation, alignment, cohesion]
        params = {
            BehaviorClass.TRANSIT: {"weights": [0.3, 0.8, 0.6], "speed": 10.0},
            BehaviorClass.ATTACK: {"weights": [0.8, 0.2, 0.1], "speed": 15.0},
            BehaviorClass.SCATTER: {"weights": [0.9, 0.1, 0.1], "speed": 12.0},
            BehaviorClass.ENCIRCLE: {"weights": [0.5, 0.7, 0.9], "speed": 8.0},
            BehaviorClass.DECOY: {"weights": [0.9, 0.1, 0.3], "speed": 7.0},
            BehaviorClass.UNKNOWN: {"weights": [0.3, 0.3, 0.3], "speed": 5.0},
        }
        return params.get(behavior, params[BehaviorClass.UNKNOWN])

    def update_flocking_forces(
        self,
        neighbor_positions: list[Vec3],
        neighbor_velocities: list[Vec3],
        swarm_centroid: Vec3 | None = None,
        dt: float = 0.1,
    ) -> None:
        """
        Update velocity based on Reynolds flocking forces.

        Parameters
        ----------
        neighbor_positions : list[Vec3]
            Positions of nearby drones.
        neighbor_velocities : list[Vec3]
            Velocities of nearby drones.
        swarm_centroid : Vec3 | None
            Center of swarm (for cohesion target).
        dt : float
            Time step.
        """
        pos = self.state.position
        vel = self.state.velocity
        params = self._behavior_params
        weights = params["weights"]
        max_speed = params["speed"]

        # Separation force
        separation = np.zeros(3, dtype=np.float64)
        min_sep_dist = 3.0
        for n_pos in neighbor_positions:
            diff = pos - n_pos
            dist = np.linalg.norm(diff)
            if 0 < dist < min_sep_dist:
                separation += diff / (dist**2 + 1e-6)

        # Alignment force
        alignment = np.zeros(3, dtype=np.float64)
        if neighbor_velocities:
            alignment = np.mean(neighbor_velocities, axis=0)

        # Cohesion force
        cohesion = np.zeros(3, dtype=np.float64)
        if neighbor_positions:
            neighbor_centroid = np.mean(neighbor_positions, axis=0)
            cohesion = neighbor_centroid - pos

        # Normalize forces
        if np.linalg.norm(separation) > 1e-6:
            separation = separation / np.linalg.norm(separation)
        if np.linalg.norm(alignment) > 1e-6:
            alignment = alignment / np.linalg.norm(alignment)
        if np.linalg.norm(cohesion) > 1e-6:
            cohesion = cohesion / np.linalg.norm(cohesion)

        # Apply weights
        force = (
            weights[0] * separation
            + weights[1] * alignment
            + weights[2] * cohesion
        )

        # Update velocity
        new_vel = vel + force * dt

        # Clamp to max speed
        speed = np.linalg.norm(new_vel)
        if speed > max_speed:
            new_vel = new_vel / speed * max_speed

        self.state.velocity = new_vel

    def step(self, dt: float = 0.1, timestamp: float | None = None) -> None:
        """
        Advance drone state by one time step.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        timestamp : float | None
            New timestamp. Uses current time if None.
        """
        # Update position: p = p + v * dt
        self.state.position = self.state.position + self.state.velocity * dt

        # Update timestamp
        if timestamp is not None:
            self.state.timestamp = timestamp

        # Update RF power based on distance from origin (simulated)
        dist = np.linalg.norm(self.state.position)
        # RF power decreases with distance (simplified path loss model)
        self.state.rf_power_dbm = -30 - 20 * np.log10(max(dist, 10)) / 10

    def set_behavior(self, behavior: BehaviorClass) -> None:
        """Change drone behavior."""
        self.state.behavior = behavior
        self._behavior_params = self._get_behavior_params(behavior)

    def apply_external_force(
        self, force: Vec3, dt: float = 0.1, mass: float = 1.0
    ) -> None:
        """
        Apply external force (e.g., jamming, interception).

        Parameters
        ----------
        force : Vec3
            Force vector in Newtons.
        dt : float
            Duration of force application.
        mass : float
            Drone mass in kg.
        """
        acceleration = force / mass
        delta_v = acceleration * dt
        self.state.velocity = self.state.velocity + delta_v
