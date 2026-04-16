"""Named simulation scenarios for demos and testing."""

from __future__ import annotations

import asyncio
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from ..constants import BehaviorClass
from .swarm_sim import SwarmSimulator

if TYPE_CHECKING:
    pass


class ScenarioType(IntEnum):
    """Available simulation scenarios."""

    SINGLE_SWARM_TRANSIT = 0
    SATURATION_ATTACK = 1
    DECOY_AND_STRIKE = 2
    ENCIRCLEMENT = 3
    MULTI_SWARM_COORDINATED = 4
    SCATTER_MANEUVER = 5


class ScenarioRunner:
    """
    Runs predefined simulation scenarios.

    Each scenario sets up specific swarm configurations
    and behavior patterns for demonstration or testing.
    """

    def __init__(self, simulator: SwarmSimulator | None = None) -> None:
        """
        Initialize the scenario runner.

        Parameters
        ----------
        simulator : SwarmSimulator | None
            Simulator instance. Created if None.
        """
        self.simulator = simulator or SwarmSimulator()
        self.current_scenario: ScenarioType | None = None

    def run_scenario(
        self,
        scenario: ScenarioType,
        seed: int | None = None,
    ) -> None:
        """
        Set up and run a specific scenario.

        Parameters
        ----------
        scenario : ScenarioType
            Scenario to run.
        seed : int | None
            Random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)

        # Clear existing swarms
        self.simulator.swarms.clear()

        self.current_scenario = scenario

        if scenario == ScenarioType.SINGLE_SWARM_TRANSIT:
            self._setup_single_swarm_transit()
        elif scenario == ScenarioType.SATURATION_ATTACK:
            self._setup_saturation_attack()
        elif scenario == ScenarioType.DECOY_AND_STRIKE:
            self._setup_decoy_and_strike()
        elif scenario == ScenarioType.ENCIRCLEMENT:
            self._setup_encirclement()
        elif scenario == ScenarioType.MULTI_SWARM_COORDINATED:
            self._setup_multi_swarm_coordinated()
        elif scenario == ScenarioType.SCATTER_MANEUVER:
            self._setup_scatter_maneuver()

    def _setup_single_swarm_transit(self) -> None:
        """
        Single swarm moving cohesively toward a target.

        Scenario: 15 drones in TRANSIT behavior.
        """
        self.simulator.create_swarm(
            swarm_id="swarm_transit_1",
            n_drones=15,
            behavior=BehaviorClass.TRANSIT,
            centroid=np.array([-80, 0, 30], dtype=np.float64),
            spread=15.0,
            initial_velocity=np.array([8, 0, 0], dtype=np.float64),
        )

    def _setup_saturation_attack(self) -> None:
        """
        High-speed attack swarm with tight formation.

        Scenario: 25 drones in ATTACK behavior, high threat.
        """
        self.simulator.create_swarm(
            swarm_id="swarm_attack_1",
            n_drones=25,
            behavior=BehaviorClass.ATTACK,
            centroid=np.array([-60, -40, 25], dtype=np.float64),
            spread=10.0,
            initial_velocity=np.array([12, 5, 0], dtype=np.float64),
        )

    def _setup_decoy_and_strike(self) -> None:
        """
        Decoy swarm (scattered) with hidden strike force.

        Scenario: Two swarms - one DECOY, one ATTACK.
        """
        # Decoy swarm - scattered, disordered
        self.simulator.create_swarm(
            swarm_id="swarm_decoy_1",
            n_drones=10,
            behavior=BehaviorClass.DECOY,
            centroid=np.array([-50, 50, 20], dtype=np.float64),
            spread=40.0,
            initial_velocity=np.array([3, -2, 0], dtype=np.float64),
        )

        # Hidden attack swarm - tight formation
        self.simulator.create_swarm(
            swarm_id="swarm_strike_1",
            n_drones=8,
            behavior=BehaviorClass.ATTACK,
            centroid=np.array([-70, -20, 15], dtype=np.float64),
            spread=8.0,
            initial_velocity=np.array([10, 3, 0], dtype=np.float64),
        )

    def _setup_encirclement(self) -> None:
        """
        Pincer movement - two swarms approaching from different angles.

        Scenario: Two swarms in ENCIRCLE behavior.
        """
        # Northern pincer
        self.simulator.create_swarm(
            swarm_id="swarm_encircle_n",
            n_drones=12,
            behavior=BehaviorClass.ENCIRCLE,
            centroid=np.array([-40, 60, 25], dtype=np.float64),
            spread=20.0,
            initial_velocity=np.array([5, -5, 0], dtype=np.float64),
        )

        # Southern pincer
        self.simulator.create_swarm(
            swarm_id="swarm_encircle_s",
            n_drones=12,
            behavior=BehaviorClass.ENCIRCLE,
            centroid=np.array([-40, -60, 25], dtype=np.float64),
            spread=20.0,
            initial_velocity=np.array([5, 5, 0], dtype=np.float64),
        )

    def _setup_multi_swarm_coordinated(self) -> None:
        """
        Multiple swarms with different behaviors coordinating.

        Scenario: Three swarms - TRANSIT, SCATTER, ATTACK.
        """
        # Lead swarm - TRANSIT
        self.simulator.create_swarm(
            swarm_id="swarm_lead",
            n_drones=10,
            behavior=BehaviorClass.TRANSIT,
            centroid=np.array([-80, 0, 30], dtype=np.float64),
            spread=15.0,
            initial_velocity=np.array([8, 0, 0], dtype=np.float64),
        )

        # Flank left - SCATTER
        self.simulator.create_swarm(
            swarm_id="swarm_flank_l",
            n_drones=8,
            behavior=BehaviorClass.SCATTER,
            centroid=np.array([-60, 40, 20], dtype=np.float64),
            spread=25.0,
            initial_velocity=np.array([10, -3, 0], dtype=np.float64),
        )

        # Flank right - ATTACK
        self.simulator.create_swarm(
            swarm_id="swarm_flank_r",
            n_drones=10,
            behavior=BehaviorClass.ATTACK,
            centroid=np.array([-60, -40, 20], dtype=np.float64),
            spread=12.0,
            initial_velocity=np.array([12, 3, 0], dtype=np.float64),
        )

    def _setup_scatter_maneuver(self) -> None:
        """
        Swarm performing saturation dispersal maneuver.

        Scenario: 20 drones in SCATTER behavior.
        """
        self.simulator.create_swarm(
            swarm_id="swarm_scatter_1",
            n_drones=20,
            behavior=BehaviorClass.SCATTER,
            centroid=np.array([-50, 0, 30], dtype=np.float64),
            spread=15.0,
            initial_velocity=np.array([8, 0, 0], dtype=np.float64),
        )

    async def run_simulation(
        self,
        n_steps: int = 100,
        dt: float = 0.1,
        callback: callable | None = None,
    ) -> list[dict]:
        """
        Run simulation for specified number of steps.

        Parameters
        ----------
        n_steps : int
            Number of simulation steps.
        dt : float
            Time step in seconds.
        callback : callable | None
            Optional callback called after each step with swarm states.

        Returns
        -------
        list[dict]
            History of swarm states.
        """
        history = []

        for _ in range(n_steps):
            swarm_states = self.simulator.step(dt)

            # Convert to serializable format
            state_snapshot = {
                "timestamp": self.simulator.timestamp,
                "swarms": {},
            }

            for swarm_id, state in swarm_states.items():
                state_snapshot["swarms"][swarm_id] = {
                    "n_drones": len(state.drones),
                    "behavior": state.behavior.name,
                    "centroid": state.centroid.tolist(),
                    "spread_radius": state.spread_radius,
                    "velocity_coherence": state.velocity_coherence,
                    "threat_score": state.threat_score,
                }

            history.append(state_snapshot)

            if callback:
                await callback(swarm_states)

            await asyncio.sleep(0)  # Yield to event loop

        return history

    def get_scenario_description(self, scenario: ScenarioType) -> str:
        """Get human-readable description of a scenario."""
        descriptions = {
            ScenarioType.SINGLE_SWARM_TRANSIT: "Single swarm in transit formation",
            ScenarioType.SATURATION_ATTACK: "High-speed attack swarm (25 drones)",
            ScenarioType.DECOY_AND_STRIKE: "Decoy swarm with hidden strike force",
            ScenarioType.ENCIRCLEMENT: "Two-swarm pincer movement",
            ScenarioType.MULTI_SWARM_COORDINATED: "Three swarms with coordinated behaviors",
            ScenarioType.SCATTER_MANEUVER: "Saturation dispersal maneuver",
        }
        return descriptions.get(scenario, "Unknown scenario")
