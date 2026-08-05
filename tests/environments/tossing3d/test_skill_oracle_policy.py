"""Offline tests for the oracle's three-way branch and its parameter provenance."""

import numpy as np
import pytest

from hitl_pmp.core.problem.tasks.types import Goal
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import GRASP_THRESHOLD
from hitl_pmp.environments.tossing3d.skill_oracle_policy import (
    ORACLE_PICK_DISTANCE,
    ORACLE_PICK_ROTATION,
    ORACLE_THROW_STANDOFF,
    SkillOraclePolicy,
)
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DOracle
from hitl_pmp.environments.tossing3d.skills import (
    PICK_DISTANCE_BOUNDS,
    PICK_ROTATION_BOUNDS,
    THROW_STANDOFF_BOUNDS,
)

from .observations import COINCIDENT_BIN_X, state

_ENV = Tossing3DEnvironment()
_EMPTY_GOAL = Goal(atoms=frozenset())


def _act(**kwargs):
    return SkillOraclePolicy.get_labeled_action(state=state(**kwargs), env=_ENV, goal=_EMPTY_GOAL)


def test_the_oracle_picks_when_the_cube_is_on_the_ground() -> None:
    action = _act()
    assert action.action[0] == pytest.approx(Tossing3DEnvironment.pick_id)
    assert action.action[1] == pytest.approx(ORACLE_PICK_DISTANCE)
    assert action.action[2] == pytest.approx(ORACLE_PICK_ROTATION)


def test_the_oracle_walks_to_the_throw_pose_once_it_is_holding_the_cube() -> None:
    action = _act(gripper=GRASP_THRESHOLD + 0.5, cube_z=0.4)
    assert action.action[0] == pytest.approx(Tossing3DEnvironment.move_to_throw_pose_id)
    assert action.action[1] == pytest.approx(ORACLE_THROW_STANDOFF)


def test_the_oracle_throws_once_it_is_holding_the_cube_and_near_the_bin() -> None:
    action = _act(
        gripper=GRASP_THRESHOLD + 0.5,
        cube_z=0.4,
        base_x=COINCIDENT_BIN_X - ORACLE_THROW_STANDOFF,
    )
    assert action.action[0] == pytest.approx(Tossing3DEnvironment.toss_id)


def test_the_oracle_solves_the_domain_in_exactly_three_skills() -> None:
    """The whole plan shape, walked symbolically: pick, walk, throw. Anything longer
    would mean `Tossing3DProblem.max_episode_steps` is under-budgeted."""
    ids = [
        _act().action[0],
        _act(gripper=GRASP_THRESHOLD + 0.5, cube_z=0.4).action[0],
        _act(
            gripper=GRASP_THRESHOLD + 0.5,
            cube_z=0.4,
            base_x=COINCIDENT_BIN_X - ORACLE_THROW_STANDOFF,
        ).action[0],
    ]
    assert ids == pytest.approx([
        Tossing3DEnvironment.pick_id,
        Tossing3DEnvironment.move_to_throw_pose_id,
        Tossing3DEnvironment.toss_id,
    ])


def test_the_oracle_offers_no_recovery_after_a_missed_toss() -> None:
    """It falls back to `Pick`, whose grasp will fail to plan, rather than to some
    invented retrieval skill. There is nothing this domain can do once the cube is past
    the barrier, and pretending otherwise would hide the irreversibility it exists to
    exhibit."""
    action = _act(cube_x=2.6, cube_z=0.025, gripper=0.0)
    assert action.action[0] == pytest.approx(Tossing3DEnvironment.pick_id)


def test_the_oracle_parameters_lie_inside_the_samplers_own_ranges() -> None:
    """An oracle drawing from outside the range a learner samples would be measuring a
    different skill from the one being learned."""
    assert PICK_DISTANCE_BOUNDS[0] <= ORACLE_PICK_DISTANCE <= PICK_DISTANCE_BOUNDS[1]
    assert PICK_ROTATION_BOUNDS[0] <= ORACLE_PICK_ROTATION <= PICK_ROTATION_BOUNDS[1]
    assert THROW_STANDOFF_BOUNDS[0] <= ORACLE_THROW_STANDOFF <= THROW_STANDOFF_BOUNDS[1]


def test_the_pick_parameters_are_upstreams_own_draw_at_its_own_rng_seed() -> None:
    """Upstream's `test_pick_ground_toss` parameterizes this grasp with
    `sample_parameters(state, np.random.default_rng(123))`. With one cube in the scene
    upstream's rejection loop has nothing to reject against, so its first draw is
    accepted and the sampler reduces to two uniforms over its own bounds -- reproduced
    here with plain numpy, so this holds without KINDER installed.
    `test_kinder_fidelity.py` checks it against the real sampler when it is."""
    rng = np.random.default_rng(123)
    assert rng.uniform(*PICK_DISTANCE_BOUNDS) == pytest.approx(ORACLE_PICK_DISTANCE)
    assert rng.uniform(*PICK_ROTATION_BOUNDS) == pytest.approx(ORACLE_PICK_ROTATION)


def test_the_throw_standoff_is_upstreams_own_test_value() -> None:
    assert ORACLE_THROW_STANDOFF == 1.35


def test_the_label_names_the_skill_its_objects_and_its_parameters() -> None:
    """`LabeledAction.label` is what the renderer burns into the frame, so it has to say
    what actually happened rather than just which skill ran."""
    label = _act().label
    assert label.startswith("Pick(robot, cube_0, cuboid_barrier, bin_0)")
    assert "params=[0.57, -0.7]" in label


def test_a_parameterless_skill_gets_no_params_suffix() -> None:
    label = _act(
        gripper=GRASP_THRESHOLD + 0.5,
        cube_z=0.4,
        base_x=COINCIDENT_BIN_X - ORACLE_THROW_STANDOFF,
    ).label
    assert "params=" not in label


def test_the_provider_forwards_its_configured_standoff() -> None:
    """The standoff that solves depends on which scene is loaded -- 1.35 on the
    coincident config, 1.55 on stock -- so it is a constructor field, not a constant."""
    oracle = Tossing3DOracle(env=_ENV, throw_standoff=1.55)
    action = oracle.get_labeled_action(
        state=state(gripper=GRASP_THRESHOLD + 0.5, cube_z=0.4), goal=_EMPTY_GOAL
    )
    assert action.action[1] == pytest.approx(1.55)
