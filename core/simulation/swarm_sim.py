"""Multi-swarm simulator for generating demo data."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..constants import BehaviorClass
from .drone import Drone, DroneState

Vec3 = npt.NDArray[np.float64]


@dataclass
class SwarmState:
    """
    Aggregate state of a drone swarm.

    Attributes
    ----------
    swarm_id : str
        Unique swarm identifier.
    drones : list[DroneState]
        List of drone states in the swarm.
    behavior : BehaviorClass
        Overall swarm behavior.
    centroid : Vec3
        Center of mass position.
    spread_radius : float
        Average distance from centroid.
    velocity_coherence : float
        How aligned velocities are (0-1).
    threat_score : float
        Computed threat score (0-1).
    """

    swarm_id: str
    drones: list[DroneState]
    behavior: BehaviorClass
    centroid: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    spread_radius: float = 0.0
    velocity_coherence: float = 0.0
    threat_score: float = 0.0

    def __post_init__(self) -> None:
        """Compute derived attributes."""
        if self.drones:
            positions = np.array([d.position for d in self.drones])
            velocities = np.array([d.velocity for d in self.drones])

            self.centroid = np.mean(positions, axis=0)
            self.spread_radius = float(
                np.mean(np.linalg.norm(positions - self.centroid, axis=1))
            )

            # Velocity coherence
            if np.all(np.linalg.norm(velocities, axis=1) > 1e-6):
                norm_vels = velocities / np.linalg.norm(velocities, axis=1, keepdims=True)
                mean_vel = np.mean(norm_vels, axis=0)
                self.velocity_coherence = float(np.linalg.norm(mean_vel))
            else:
                self.velocity_coherence = 0.0

            # Threat score based on behavior
            threat_map = {
                BehaviorClass.ATTACK: 0.95,
                BehaviorClass.SCATTER: 0.80,
                BehaviorClass.ENCIRCLE: 0.85,
                BehaviorClass.DECOY: 0.40,
                BehaviorClass.TRANSIT: 0.50,
                BehaviorClass.UNKNOWN: 0.30,
            }
            self.threat_score = threat_map.get(self.behavior, 0.30)


class SwarmSimulator:
    """
    Multi-swarm simulation engine.

    Manages multiple drone swarms with independent behaviors
    and generates synthetic sensor data.

    Attributes
    ----------
    swarms : dict[str, list[Drone]]
        Active swarms keyed by swarm_id.
    timestamp : float
        Current simulation timestamp.
    running : bool
        Whether simulation is active.
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize the simulator.

        Parameters
        ----------
        seed : int | None
            Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)

        self.swarms: dict[str, list[Drone]] = {}
        self.timestamp = time.time()
        self.running = False
        self._step_count = 0

    def create_swarm(
        self,
        swarm_id: str,
        n_drones: int = 10,
        behavior: BehaviorClass = BehaviorClass.TRANSIT,
        centroid: Vec3 | None = None,
        spread: float = 20.0,
        initial_velocity: Vec3 | None = None,
    ) -> list[Drone]:
        """
        Create a new swarm of drones.

        Parameters
        ----------
        swarm_id : str
            Unique swarm identifier.
        n_drones : int
            Number of drones in the swarm.
        behavior : BehaviorClass
            Initial swarm behavior.
        centroid : Vec3 | None
            Starting centroid position. Random if None.
        spread : float
            Initial spread radius in meters.
        initial_velocity : Vec3 | None
            Common initial velocity. Zero if None.

        Returns
        -------
        list[Drone]
            Created drone instances.
        """
        if centroid is None:
            centroid = np.random.uniform(-50, 50, 3).astype(np.float64)

        drones: list[Drone] = []

        for i in range(n_drones):
            # Random position around centroid
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, spread)
            offset = np.array(
                [
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    np.random.uniform(-5, 5),
                ],
                dtype=np.float64,
            )

            position = centroid + offset

            if initial_velocity is None:
                # Small random velocity
                velocity = np.random.uniform(-1, 1, 3).astype(np.float64)
            else:
                velocity = initial_velocity.astype(np.float64)

            drone = Drone(
                drone_id=f"{swarm_id}_drone_{i}",
                swarm_id=swarm_id,
                initial_position=position,
                initial_velocity=velocity,
                behavior=behavior,
            )

            drones.append(drone)

        self.swarms[swarm_id] = drones
        return drones

    def remove_swarm(self, swarm_id: str) -> None:
        """Remove a swarm from simulation."""
        if swarm_id in self.swarms:
            del self.swarms[swarm_id]

    def step(self, dt: float = 0.1) -> dict[str, SwarmState]:
        """
        Advance simulation by one time step.

        Parameters
        ----------
        dt : float
            Time step in seconds.

        Returns
        -------
        dict[str, SwarmState]
            State of all swarms after the step.
        """
        self.timestamp += dt
        self._step_count += 1

        swarm_states: dict[str, SwarmState] = {}

        for swarm_id, drones in self.swarms.items():
            # Update each drone
            for drone in drones:
                if not drone.state.is_alive:
                    continue

                # Find neighbors for flocking
                neighbors = [
                    d for d in drones
                    if d.state.drone_id != drone.state.drone_id and d.state.is_alive
                ]

                if neighbors:
                    neighbor_positions = [d.state.position for d in neighbors]
                    neighbor_velocities = [d.state.velocity for d in neighbors]

                    drone.update_flocking_forces(
                        neighbor_positions,
                        neighbor_velocities,
                        swarm_centroid=None,
                        dt=dt,
                    )

                drone.step(dt=dt, timestamp=self.timestamp)

            # Compute swarm state
            drone_states = [d.state for d in drones if d.state.is_alive]
            if drone_states:
                # Determine dominant behavior
                behavior_counts: dict[BehaviorClass, int] = {}
                for ds in drone_states:
                    behavior_counts[ds.behavior] = behavior_counts.get(ds.behavior, 0) + 1
                dominant_behavior = max(behavior_counts, key=behavior_counts.get)

                swarm_states[swarm_id] = SwarmState(
                    swarm_id=swarm_id,
                    drones=drone_states,
                    behavior=dominant_behavior,
                )

        return swarm_states

    def get_all_drones(self) -> list[DroneState]:
        """Get state of all drones across all swarms."""
        all_drones: list[DroneState] = []
        for drones in self.swarms.values():
            all_drones.extend([d.state for d in drones if d.state.is_alive])
        return all_drones

    def get_detections(self) -> list[list[float]]:
        """
        Get current detections for tracking system.

        Returns
        -------
        list[list[float]]
            List of [px, py, pz] detections with simulated noise.
        """
        detections = []
        for drone in self.get_all_drones():
            # Add measurement noise
            noise = np.random.normal(0, 0.5, 3)
            detection = drone.position + noise
            detections.append(detection.tolist())
        return detections

    def set_swarm_behavior(
        self, swarm_id: str, behavior: BehaviorClass
    ) -> None:
        """Change behavior of all drones in a swarm."""
        if swarm_id in self.swarms:
            for drone in self.swarms[swarm_id]:
                drone.set_behavior(behavior)

    def apply_swarm_force(
        self, swarm_id: str, force: Vec3, dt: float = 0.1
    ) -> None:
        """Apply external force to all drones in a swarm."""
        if swarm_id in self.swarms:
            for drone in self.swarms[swarm_id]:
                drone.apply_external_force(force, dt)

    def get_statistics(self) -> dict:
        """Get simulation statistics."""
        total_drones = sum(len(drones) for drones in self.swarms.values())
        active_drones = sum(
            sum(1 for d in drones if d.state.is_alive)
            for drones in self.swarms.values()
        )

        return {
            "n_swarms": len(self.swarms),
            "total_drones": total_drones,
            "active_drones": active_drones,
            "timestamp": self.timestamp,
            "step_count": self._step_count,
        }
