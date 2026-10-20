"""The always-ask reset wrapper on SweepOffShelf3D: the cost file that prices its full
reset, and the reset restoring a whole start scene. Which skills owe a reset, and that the
stranded trigger cannot fire here, are pinned off-simulator in `test_symbolic.py`."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.core.practice_costs import HumanCharge, PracticeCosts
from hitl_pmp.core.problem.human.types import (
    SAMPLED_START_RESET_DESTINATION,
    SAMPLED_START_RESET_SKILL_NAME,
)
from hitl_pmp.environments.sweepoffshelf3d.cli import SweepOffShelf3DCli
from hitl_pmp.environments.sweepoffshelf3d.environment import (
    SKILL_IDS,
    SweepOffShelf3DEnvironment,
)
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects
from hitl_pmp.environments.sweepoffshelf3d.step_protocol import SweepOffShelf3DStepProtocol

_NEEDS_KINDER = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator dependency"
)


def test_the_shelf_sampled_start_cost_file_prices_the_full_reset_like_the_shelf_resets():
    """Under step accounting the full reset is charged by name, and an unpriced name
    raises at the first reset -- the shelf's own cost files price only `reset_cubes` and
    `reset_wiper`. This file adds the full reset at the price both carry in the
    canonical shelf runs (cost 1, one counted step) and changes nothing else."""
    root = Path(__file__).resolve().parents[3]
    tossing = PracticeCosts.load(path=root / "configs" / "normalized_practice_costs.json")
    costs = PracticeCosts.load(
        path=root / "configs" / "sweepoffshelf3d_practice_costs_sampled_start.json"
    )
    shelf_skills = set(SweepOffShelf3DStepProtocol.human_skill_names().values())
    assert set(costs.human_skills) == {*shelf_skills, SAMPLED_START_RESET_SKILL_NAME}
    for charge in costs.human_skills.values():
        assert charge == HumanCharge()
    assert costs.model_dump(exclude={"human_skills"}) == tossing.model_dump(
        exclude={"human_skills"}
    )


_START_ATOMS = frozenset(
    {("OnShelf", (cube.name,)) for cube in SweepOffShelf3DObjects.cubes}
    | {("WiperHome", ("wiper_0",)), ("HandEmpty", ("robot",)), ("AnyCubeOnShelf", ())}
)


@_NEEDS_KINDER
def test_a_full_reset_restores_every_cube_the_wiper_and_the_robot_to_a_start_scene() -> None:
    """From a scene the robot has disturbed -- the wiper in hand, the base moved -- and
    a cube knocked to the floor, one full reset gives the shelf's start: all five cubes
    on the shelf, the wiper home, the hand empty, and the robot exactly where a fresh
    build of the drawn scene puts it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    SweepOffShelf3DCli.add_arguments(parser=parser)
    problem = SweepOffShelf3DCli.build_problem(
        args=parser.parse_args(["--no-scene-bg"]), human_seed=0
    )
    env = problem.env
    assert isinstance(env, SweepOffShelf3DEnvironment)
    robot = SweepOffShelf3DObjects.robot
    scene = SweepOffShelf3DObjects.scene
    try:
        problem.reset_to_task(task=problem.sample_train_task())
        start_robot = env.get_current_state()[robot].copy()
        env.take_action(action=np.array([float(SKILL_IDS["PickWiper"]), 0.0, 0.0, 0.0]))
        cube = env.backend().core().get_object("cube_0")
        cube.set_pose([0.0, 0.0, 0.05], [1.0, 0.0, 0.0, 0.0])
        cube.set_velocity([0.0] * 3, [0.0] * 3)
        for _ in range(500):
            env.backend().sim().step()
        disturbed = env.backend().abstract_atoms()
        assert ("HoldingWiper", ("robot", "wiper_0")) in disturbed
        assert ("OnShelf", ("cube_0",)) not in disturbed
        assert not np.allclose(env.get_current_state()[robot], start_robot)

        problem.execute_movables_reset(destination=SAMPLED_START_RESET_DESTINATION)

        state = env.get_current_state()
        assert env.backend().abstract_atoms() == _START_ATOMS
        assert int(round(state.get(obj=scene, feature_name="steps_taken"))) == 0
        seed = int(round(state.get(obj=scene, feature_name="seed")))
        rebuilt = env.reset_to_seed(seed=seed)
        assert np.allclose(state[robot], rebuilt[robot])
        for obj in (*SweepOffShelf3DObjects.cubes, SweepOffShelf3DObjects.wiper):
            assert np.allclose(state[obj], rebuilt[obj])
    finally:
        env.close()
