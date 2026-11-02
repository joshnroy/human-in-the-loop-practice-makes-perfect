"""A correct pour must not be followed by a failed fixed pitcher return."""

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KinDER not installed"
)


def test_native_wood_one_cube_pour_returns_pitcher_within_existing_home_tolerance():
    from kinder.envs.dynamic3d.pitcher_pour3d import PitcherPour3DEnv

    from hitl_pmp.environments.pouring3d.lateral_environment import LateralPouringEnvironment

    task = (
        Path(inspect.getfile(PitcherPour3DEnv)).parent
        / "tasks/PitcherPour3D/PitcherPour3D-o5-winco.json"
    )
    env = LateralPouringEnvironment(
        seed=0, requested_count=1, cube_material="wood", scene_bg=True, task_config_path=str(task)
    )
    env.hard_reset()
    try:
        env.take_action(action=np.array([0, 0, 95.5, 2.5]))
        assert env.last_info["success"]
        env.take_action(action=np.array([1, 0, 95.5, 2.5]))
        # This checks placement after a physically executed pour. Exact-count
        # success is a separate controller outcome and varies across platforms.
        assert env.last_info["error"] is None, env.last_info
        assert env.sim.held == "pitcher"
        before = env.sim.locations()
        env.take_action(action=np.array([2, 0, 95.5, 2.5]))
        assert env.last_info["success"], env.last_info["after"]["bodies"]["pitcher"]
        assert env.sim.controller.vessel_home(name="pitcher")
        assert env.sim.held is None
        assert env.sim.locations() == before
        assert env.sim.controller.scene._check_goals() == (
            before.count("tray") == 1
            and before.count("pitcher") == 4
            and env.sim.controller.scene.cubes_settled()
        )
    finally:
        env.sim.close()
