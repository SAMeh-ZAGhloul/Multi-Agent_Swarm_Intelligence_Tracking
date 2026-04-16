"""Test decoy and strike scenario."""

import numpy as np
import pytest

from core.simulation import ScenarioRunner, ScenarioType
from core.constants import BehaviorClass


class TestDecoyAndStrike:
    """Test decoy and strike scenario."""

    def test_scenario_setup(self) -> None:
        """Test scenario initialization."""
        runner = ScenarioRunner()
        runner.run_scenario(ScenarioType.DECOY_AND_STRIKE)

        assert runner.current_scenario == ScenarioType.DECOY_AND_STRIKE
        assert "swarm_decoy_1" in runner.simulator.swarms
        assert "swarm_strike_1" in runner.simulator.swarms

    def test_drone_count(self) -> None:
        """Test correct number of drones."""
        runner = ScenarioRunner()
        runner.run_scenario(ScenarioType.DECOY_AND_STRIKE)

        drones = runner.simulator.get_all_drones()

        # 10 decoy + 8 strike = 18 total
        assert len(drones) == 18

    def test_two_swarms(self) -> None:
        """Test two distinct swarms exist."""
        runner = ScenarioRunner()
        runner.run_scenario(ScenarioType.DECOY_AND_STRIKE)

        states = runner.simulator.step(0.1)

        assert len(states) == 2
        assert "swarm_decoy_1" in states
        assert "swarm_strike_1" in states

    def test_decoy_behavior(self) -> None:
        """Test decoy swarm behavior."""
        runner = ScenarioRunner()
        runner.run_scenario(ScenarioType.DECOY_AND_STRIKE)

        states = runner.simulator.step(0.1)

        decoy_state = states["swarm_decoy_1"]
        assert decoy_state.behavior == BehaviorClass.DECOY

    def test_strike_behavior(self) -> None:
        """Test strike swarm behavior."""
        runner = ScenarioRunner()
        runner.run_scenario(ScenarioType.DECOY_AND_STRIKE)

        states = runner.simulator.step(0.1)

        strike_state = states["swarm_strike_1"]
        assert strike_state.behavior == BehaviorClass.ATTACK

    def test_threat_differentiation(self) -> None:
        """Test strike has higher threat than decoy."""
        runner = ScenarioRunner()
        runner.run_scenario(ScenarioType.DECOY_AND_STRIKE)

        states = runner.simulator.step(0.1)

        decoy_threat = states["swarm_decoy_1"].threat_score
        strike_threat = states["swarm_strike_1"].threat_score

        # Strike should be higher threat
        assert strike_threat > decoy_threat

    def test_spatial_separation(self) -> None:
        """Test swarms are spatially separated."""
        runner = ScenarioRunner()
        runner.run_scenario(ScenarioType.DECOY_AND_STRIKE)

        states = runner.simulator.step(0.1)

        decoy_centroid = states["swarm_decoy_1"].centroid
        strike_centroid = states["swarm_strike_1"].centroid

        distance = np.linalg.norm(decoy_centroid - strike_centroid)

        # Should be significantly separated
        assert distance > 20.0  # At least 20 meters apart

    def test_decoy_spread(self) -> None:
        """Test decoy has larger spread than strike."""
        runner = ScenarioRunner()
        runner.run_scenario(ScenarioType.DECOY_AND_STRIKE)

        states = runner.simulator.step(0.1)

        decoy_spread = states["swarm_decoy_1"].spread_radius
        strike_spread = states["swarm_strike_1"].spread_radius

        # Decoy should be more spread out
        assert decoy_spread > strike_spread
