"""Reynolds flocking model and parameter inversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..constants import (
    NEIGHBORHOOD_RADIUS,
    REYNOLDS_WEIGHT_ALIGNMENT,
    REYNOLDS_WEIGHT_COHESION,
    REYNOLDS_WEIGHT_SEPARATION,
)

Vec3 = npt.NDArray[np.float64]
Mat3x3 = npt.NDArray[np.float64]


@dataclass
class ReynoldsWeights:
    """Weights for Reynolds flocking forces."""

    separation: float = REYNOLDS_WEIGHT_SEPARATION
    alignment: float = REYNOLDS_WEIGHT_ALIGNMENT
    cohesion: float = REYNOLDS_WEIGHT_COHESION


@dataclass
class ReynoldsForces:
    """Computed flocking forces for a single agent."""

    separation: Vec3
    alignment: Vec3
    cohesion: Vec3
    total: Vec3


def compute_reynolds_forces(
    position: Vec3,
    velocity: Vec3,
    neighbor_positions: list[Vec3],
    neighbor_velocities: list[Vec3],
    weights: ReynoldsWeights | None = None,
    neighborhood_radius: float = NEIGHBORHOOD_RADIUS,
    min_distance: float = 2.0,
) -> ReynoldsForces:
    """
    Compute Reynolds flocking forces for a single agent.

    Parameters
    ----------
    position : Vec3
        Current agent position.
    velocity : Vec3
        Current agent velocity.
    neighbor_positions : list[Vec3]
        Positions of neighbors within radius.
    neighbor_velocities : list[Vec3]
        Velocities of neighbors.
    weights : ReynoldsWeights | None
        Force weights. Uses defaults if None.
    neighborhood_radius : float
        Maximum distance to consider neighbors.
    min_distance : float
        Minimum separation distance.

    Returns
    -------
    forces : ReynoldsForces
        Computed separation, alignment, cohesion, and total forces.
    """
    if weights is None:
        weights = ReynoldsWeights()

    separation = np.zeros(3, dtype=np.float64)
    alignment = np.zeros(3, dtype=np.float64)
    cohesion = np.zeros(3, dtype=np.float64)

    neighbor_count = 0

    for n_pos, n_vel in zip(neighbor_positions, neighbor_velocities, strict=False):
        # Vector to neighbor
        diff = n_pos - position
        distance = np.linalg.norm(diff)

        # Skip if outside neighborhood
        if distance > neighborhood_radius or distance < 1e-6:
            continue

        neighbor_count += 1

        # Separation: steer away from close neighbors
        if distance < min_distance:
            separation += -diff / (distance + 1e-6)

        # Alignment: match velocity with neighbors
        alignment += n_vel

        # Cohesion: steer toward average position
        cohesion += diff

    # Normalize by neighbor count
    if neighbor_count > 0:
        alignment /= neighbor_count
        cohesion /= neighbor_count

        # Cohesion: normalize and scale by distance
        cohesion_norm = np.linalg.norm(cohesion)
        if cohesion_norm > 1e-6:
            cohesion = cohesion / cohesion_norm * min(cohesion_norm, 10.0)

    # Normalize separation
    sep_norm = np.linalg.norm(separation)
    if sep_norm > 1e-6:
        separation = separation / sep_norm

    # Apply weights
    separation *= weights.separation
    alignment *= weights.alignment
    cohesion *= weights.cohesion

    # Total force
    total = separation + alignment + cohesion

    return ReynoldsForces(
        separation=separation,
        alignment=alignment,
        cohesion=cohesion,
        total=total,
    )


class ReynoldsFlocking:
    """
    Multi-agent Reynolds flocking simulator.

    This class manages a swarm of agents and computes flocking behavior
    for all agents simultaneously.

    Attributes
    ----------
    positions : npt.NDArray
        Agent positions, shape (n_agents, 3).
    velocities : npt.NDArray
        Agent velocities, shape (n_agents, 3).
    weights : ReynoldsWeights
        Current flocking weights.
    """

    def __init__(
        self,
        n_agents: int,
        positions: npt.NDArray[np.float64] | None = None,
        velocities: npt.NDArray[np.float64] | None = None,
        weights: ReynoldsWeights | None = None,
    ) -> None:
        """
        Initialize the flocking simulation.

        Parameters
        ----------
        n_agents : int
            Number of agents in the swarm.
        positions : npt.NDArray | None
            Initial positions. If None, random initialization.
        velocities : npt.NDArray | None
            Initial velocities. If None, zeros.
        weights : ReynoldsWeights | None
            Flocking weights. Uses defaults if None.
        """
        self.n_agents = n_agents
        self.weights = weights if weights else ReynoldsWeights()

        if positions is None:
            # Random positions in a 100x100x50 volume
            self.positions = np.random.uniform(
                -50, 50, size=(n_agents, 3)
            ).astype(np.float64)
        else:
            self.positions = np.asarray(positions, dtype=np.float64)

        if velocities is None:
            self.velocities = np.zeros((n_agents, 3), dtype=np.float64)
        else:
            self.velocities = np.asarray(velocities, dtype=np.float64)

    def compute_all_forces(
        self,
    ) -> npt.NDArray[np.float64]:
        """
        Compute flocking forces for all agents.

        Returns
        -------
        forces : npt.NDArray
            Force vectors, shape (n_agents, 3).
        """
        forces = np.zeros_like(self.positions)

        for i in range(self.n_agents):
            # Find neighbors within radius
            neighbor_indices = self._find_neighbors(i)

            if len(neighbor_indices) == 0:
                continue

            n_positions = self.positions[neighbor_indices]
            n_velocities = self.velocities[neighbor_indices]

            flock_forces = compute_reynolds_forces(
                self.positions[i],
                self.velocities[i],
                list(n_positions),
                list(n_velocities),
                self.weights,
            )

            forces[i] = flock_forces.total

        return forces

    def _find_neighbors(
        self, agent_idx: int
    ) -> npt.NDArray[np.int64]:
        """Find indices of neighbors within radius."""
        pos = self.positions[agent_idx]
        distances = np.linalg.norm(self.positions - pos, axis=1)

        # Exclude self and agents outside radius
        mask = (distances > 1e-6) & (distances < NEIGHBORHOOD_RADIUS)
        return np.where(mask)[0]

    def step(self, dt: float = 0.1, max_speed: float = 15.0) -> None:
        """
        Advance simulation by one time step.

        Parameters
        ----------
        dt : float
            Time step in seconds.
        max_speed : float
            Maximum agent speed in m/s.
        """
        forces = self.compute_all_forces()

        # Update velocities: v = v + F * dt
        self.velocities += forces * dt

        # Clamp speed
        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        mask = speeds > max_speed
        self.velocities = np.where(
            mask,
            self.velocities / speeds * max_speed,
            self.velocities,
        )

        # Update positions: p = p + v * dt
        self.positions += self.velocities * dt

    def get_swarm_metrics(self) -> dict:
        """Compute swarm-level metrics."""
        centroid = np.mean(self.positions, axis=0)
        distances_from_centroid = np.linalg.norm(
            self.positions - centroid, axis=1
        )
        spread_radius = float(np.mean(distances_from_centroid))

        mean_velocity = np.mean(self.velocities, axis=0)
        velocity_coherence = float(
            np.mean(
                np.sum(self.velocities * mean_velocity, axis=1)
                / (np.linalg.norm(self.velocities, axis=1) * np.linalg.norm(mean_velocity) + 1e-6)
            )
        )

        return {
            "centroid": centroid,
            "spread_radius": spread_radius,
            "velocity_coherence": velocity_coherence,
            "mean_speed": float(np.mean(np.linalg.norm(self.velocities, axis=1))),
        }
