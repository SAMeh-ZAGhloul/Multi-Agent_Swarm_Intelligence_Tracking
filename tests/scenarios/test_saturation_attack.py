"""Test saturation attack scenario."""

import numpy as np
import pytest

from core.simulation import ScenarioRunner, ScenarioType
from core.constants import BehaviorClass


class TestSaturationAttack:
    """Test saturation attack scenario."""

    def test_scenario_setup(self) -> None:
        """Test scenario initialization."""
        runner = ScenarioRunner()
        runner.run_scenario(ScenarioType.SATURATION_ATTACK)

        assert runner.current_scenario == ScenarioType.SATURATION_ATTACK
        assert "swarm_attack_1" in runner.simulator.swarms

    def test_drone_count(self) -> None:
        """Test correct number of drones."""
        runner = ScenarioRunner()
        runner.run_scenario(ScenarioType.SATURATION_ATTACK)

        drones = runner.simulator.get_all_drones()

        # Should have 25 drones
        assert len(drones) == 25

    def test_initial_behavior(self) -> None:
        """Test initial swarm behavior."""
        runner = ScenarioRunner()
        runner.run_scenario(ScenarioType.SATURATION_ATTACK)

        # Step once to compute swarm state
        states = runner.simulator.step(0.1)

        assert "swarm_attack_1" in states
        assert states["swarm_attack_1"].behavior == BehaviorClass.ATTACK

    def test_threat_score(self) -> None:
        """Test threat score is high for attack swarm."""
        runner = ScenarioRunner()
        runner.run_scenario(ScenarioType.SATURATION_ATTACK)

        states = runner.simulator.step(0.1)

        swarm_state = states["swarm_attack_1"]
        assert swarm_state.threat_score > 0.8

    def test_swarm_movement(self) -> None:
        """Test swarm moves cohesively."""
        runner = ScenarioRunner()
        runner.run_scenario(ScenarioType.SATURATION_ATTACK)

        # Get initial centroid
        states1 = runner.simulator.step(0.1)
        centroid1 = states1["swarm_attack_1"].centroid.copy()

        # Step forward
        runner.simulator.step(1.0)
        states2 = runner.simulator.step(0.1)
        centroid2 = states2["swarm_attack_1"].centroid

        # Centroid should have moved
        displacement = np.linalg.norm(centroid2 - centroid1)
        assert displacement > 1.0  # Should have moved at least 1 meter

    def test_velocity_coherence(self) -> None:
        """Test attack swarm has high velocity coherence."""
        runner = ScenarioRunner()
        runner.run_scenario(ScenarioType.SATURATION_ATTACK)

        states = runner.simulator.step(0.1)

        # Attack swarm should have coherent velocities
        coherence = states["swarm_attack_1"].velocity_coherence
        assert coherence > 0.5  # Reasonably coherent

    def test_tight_formation(self) -> None:
        """Test attack swarm maintains tight formation."""
        runner = ScenarioRunner()
        runner.run_scenario(ScenarioType.SATURATION_ATTACK)

        states = runner.simulator.step(0.1)

        # Attack swarm should have small spread
        spread = states["swarm_attack_1"].spread_radius
        assert spread < 20.0  # Tight formation

    def test_simulation_steps(self) -> None:
        """Test multiple simulation steps."""
        runner = ScenarioRunner()
        runner.run_scenario(ScenarioType.SATURATION_ATTACK)

        # Run 100 steps
        for _ in range(100):
            states = runner.simulator.step(0.1)

        # Should still have all drones
        drones = runner.simulator.get_all_drones()
        assert len(drones) == 25

        # Should still be one swarm
        assert len(states) == 1

    def test_statistics(self) -> None:
        """Test simulation statistics."""
        runner = ScenarioRunner()
        runner.run_scenario(ScenarioType.SATURATION_ATTACK)

        runner.simulator.step(0.1)

        stats = runner.simulator.get_statistics()

        assert stats["n_swarms"] == 1
        assert stats["total_drones"] == 25
        assert stats["active_drones"] == 25
        assert stats["step_count"] == 1


@pytest.mark.scenario
def test_full_scenario_run() -> None:
    """Run full saturation attack scenario end-to-end."""
    import asyncio
    from core.simulation import ScenarioRunner, ScenarioType

    runner = ScenarioRunner()
    runner.run_scenario(ScenarioType.SATURATION_ATTACK)

    async def run_and_verify() -> None:
        """Run simulation and verify behavior."""
        history = await runner.run_simulation(n_steps=50, dt=0.1)

        # Should have history for all steps
        assert len(history) == 50

        # All steps should have the swarm
        for step in history:
            assert "swarm_attack_1" in step["swarms"]

    asyncio.run(run_and_verify())
