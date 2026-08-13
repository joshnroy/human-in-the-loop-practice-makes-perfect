"""Offline tests for the oracle's three-way branch and its parameter provenance."""

import json
import pathlib

import numpy as np
import pytest

from hitl_pmp.core.problem.tasks.types import Goal
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    GRASP_THRESHOLD,
    TOSS_RELEASE_MS_BOUNDS,
    TOSS_SPEED_BOUNDS,
    UPSTREAM_DEFAULT_GRIPPER_RELEASE_MS,
    UPSTREAM_DEFAULT_RELEASE_SPEED_DEG_S,
)
from hitl_pmp.environments.tossing3d.skill_oracle_policy import (
    ORACLE_GRIPPER_RELEASE_MS,
    ORACLE_PICK_DISTANCE,
    ORACLE_PICK_ROTATION,
    ORACLE_RELEASE_SPEED_DEG_S,
    ORACLE_THROW_STANDOFF,
    SkillOraclePolicy,
)
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DOracle
from hitl_pmp.environments.tossing3d.skills import (
    PICK_DISTANCE_BOUNDS,
    PICK_ROTATION_BOUNDS,
    THROW_STANDOFF_BOUNDS,
)

from .observations import BIN_X, state

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
        base_x=BIN_X - ORACLE_THROW_STANDOFF,
    )
    assert action.action[0] == pytest.approx(Tossing3DEnvironment.toss_id)
    assert action.action[1] == pytest.approx(ORACLE_RELEASE_SPEED_DEG_S)
    assert action.action[2] == pytest.approx(ORACLE_GRIPPER_RELEASE_MS)


def test_the_oracle_solves_the_domain_in_exactly_three_skills() -> None:
    """The whole plan shape, walked symbolically: pick, walk, throw. Anything longer
    would mean `Tossing3DProblem.max_episode_steps` is under-budgeted."""
    ids = [
        _act().action[0],
        _act(gripper=GRASP_THRESHOLD + 0.5, cube_z=0.4).action[0],
        _act(
            gripper=GRASP_THRESHOLD + 0.5,
            cube_z=0.4,
            base_x=BIN_X - ORACLE_THROW_STANDOFF,
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
    assert TOSS_SPEED_BOUNDS[0] <= ORACLE_RELEASE_SPEED_DEG_S <= TOSS_SPEED_BOUNDS[1]
    assert TOSS_RELEASE_MS_BOUNDS[0] <= ORACLE_GRIPPER_RELEASE_MS <= TOSS_RELEASE_MS_BOUNDS[1]


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


def test_the_oracle_release_speed_is_upstreams_own_shipped_default() -> None:
    """140 deg/s is not a tuned number. It is the literal that was inline in
    `TossController.reset` before kb#8 made it a parameter, it is what
    `toss_profile_limits()` still returns by default, and it is the speed every committed
    Tossing3D number -- including the `10/10` at standoff 1.35 -- was measured at.

    Pinning it here is what keeps the oracle's throw *byte-identical* to the throw it
    made before the dial existed. If this number ever has to move, that is a new
    measurement, not a tweak.
    """
    assert ORACLE_RELEASE_SPEED_DEG_S == 140.0
    assert ORACLE_RELEASE_SPEED_DEG_S == UPSTREAM_DEFAULT_RELEASE_SPEED_DEG_S


def test_the_oracle_release_ms_is_neither_upstreams_nor_the_robots() -> None:
    """Upstream's 720 ms is where the retired `_release_fraction = 0.46` trigger fell.
    The robot's own 600 ms is 0.4107 of its 1476 ms swing but 0.3449 of this one."""
    assert ORACLE_GRIPPER_RELEASE_MS == 792.0
    assert ORACLE_GRIPPER_RELEASE_MS != UPSTREAM_DEFAULT_GRIPPER_RELEASE_MS
    assert UPSTREAM_DEFAULT_GRIPPER_RELEASE_MS == 720.0
    assert ORACLE_GRIPPER_RELEASE_MS != 600.0


def _measured_solving_band_ms() -> tuple[list[float], dict[float, int]]:
    """Release milliseconds that solve `5/5` at the oracle's speed, read out of
    PR #240's committed grid rather than retyped."""
    surface = (
        pathlib.Path(__file__).resolve().parents[3]
        / "docs"
        / "experiment-logs"
        / "2026-08-13-tossing3d-toss-parameter-surface.json"
    )
    solved_by_ms: dict[float, int] = {}
    for row in json.loads(surface.read_text())["rows"]:
        if row["commanded_speed_deg"] != ORACLE_RELEASE_SPEED_DEG_S:
            continue
        release_ms = row["commanded_release_ms"]
        solved_by_ms[release_ms] = solved_by_ms.get(release_ms, 0) + int(row["solved"])
    band = sorted(ms for ms, solved in solved_by_ms.items() if solved == 5)
    return band, solved_by_ms


def test_the_oracle_release_ms_is_the_midpoint_of_the_measured_solving_band() -> None:
    """PR #240's grid solves `5/5` on exactly two adjacent release milliseconds at this
    speed; `KinderBackend.run_toss` rounds to whole ms, so the midpoint is 792."""
    band, _ = _measured_solving_band_ms()
    assert len(band) == 2, "the 5/5 band at 140 deg/s is two grid cells wide"
    assert round((band[0] + band[-1]) / 2.0) == ORACLE_GRIPPER_RELEASE_MS
    assert band[0] < ORACLE_GRIPPER_RELEASE_MS < band[-1]
    assert band[0] > UPSTREAM_DEFAULT_GRIPPER_RELEASE_MS


def test_the_oracle_release_ms_clears_the_nearest_measured_failure_by_80ms() -> None:
    """The grid steps 57.9 ms: this is the measured margin, the real one >= 28.8 ms."""
    band, solved_by_ms = _measured_solving_band_ms()
    below = max(ms for ms in solved_by_ms if ms < band[0] and solved_by_ms[ms] < 5)
    above = min(ms for ms in solved_by_ms if ms > band[-1] and solved_by_ms[ms] < 5)
    assert ORACLE_GRIPPER_RELEASE_MS - below >= 80.0
    assert above - ORACLE_GRIPPER_RELEASE_MS >= 80.0
    # Upstream's 720 ms clears the same failure by under 15 ms.
    assert UPSTREAM_DEFAULT_GRIPPER_RELEASE_MS - below < 15.0


def test_the_label_names_the_skill_its_objects_and_its_parameters() -> None:
    """`LabeledAction.label` is what the renderer burns into the frame, so it has to say
    what actually happened rather than just which skill ran."""
    label = _act().label
    assert label.startswith("Pick(robot, cube_0, cuboid_barrier, bin_0)")
    assert "params=[0.57, -0.7]" in label


def test_the_toss_label_now_carries_its_release_speed() -> None:
    """`Toss` used to be this domain's one parameterless skill, so its label had no
    `params=` suffix. It has a dial now, and the renderer burns the label into the frame
    -- a clip of a throw has to say how hard the throw was, or two clips at different
    speeds are indistinguishable."""
    label = _act(
        gripper=GRASP_THRESHOLD + 0.5,
        cube_z=0.4,
        base_x=BIN_X - ORACLE_THROW_STANDOFF,
    ).label
    assert label.startswith("Toss(robot, cube_0, bin_0, cuboid_barrier)")
    assert "params=[140.0, 792.0]" in label


def test_the_provider_forwards_its_configured_standoff() -> None:
    """Which standoff solves is a property of the scene's geometry rather than a constant
    of this domain -- 1.35 on the scene as shipped today, 1.55 on the pre-#126 one whose
    bin sat 23 cm further out -- so it is a constructor field, not a constant."""
    oracle = Tossing3DOracle(env=_ENV, throw_standoff=1.55)
    action = oracle.get_labeled_action(
        state=state(gripper=GRASP_THRESHOLD + 0.5, cube_z=0.4), goal=_EMPTY_GOAL
    )
    assert action.action[1] == pytest.approx(1.55)
