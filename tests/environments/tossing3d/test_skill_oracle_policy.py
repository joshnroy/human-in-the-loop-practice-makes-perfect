"""Offline tests for the oracle's two-way branch and its parameter provenance.

**It was a three-way branch on `Holding` and `RobotAtSuccessfulThrowPose`.** Upstream
composed the base move and the throw into one controller, so there is nothing left to
choose between once the cube is held, and the branch is on `Holding` alone. Two families
of test went with the middle rung and are not ported:

- **`ORACLE_PICK_DISTANCE` / `ORACLE_PICK_ROTATION` and their provenance**, which pinned
  the pair upstream's `PickShelfController.sample_parameters` drew from
  `np.random.default_rng(123)`. `PickCube` has `param_dim=0` and derives both internally,
  so there is no oracle knowledge left to check.
- **The `MoveToThrowPose` rung** -- the "walks to the throw pose" branch and the
  `THROW_STANDOFF_BOUNDS` containment that went with it.
"""

import json
import pathlib

import pytest

from hitl_pmp.core.problem.tasks.types import Goal
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_oracle_policy import (
    ORACLE_GRIPPER_RELEASE_MS,
    ORACLE_RELEASE_SPEED_DEG_S,
    ORACLE_THROW_ROTATION,
    ORACLE_THROW_STANDOFF,
    SkillOraclePolicy,
)
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DOracle
from hitl_pmp.environments.tossing3d.skills import (
    TOSS_DISTANCE_BOUNDS,
    TOSS_RELEASE_MS_BOUNDS,
    TOSS_ROTATION_BOUNDS,
    TOSS_SPEED_BOUNDS,
)

from .observations import HOLDING_ATOMS, INITIAL_ATOMS, state

_ENV = Tossing3DEnvironment()
_EMPTY_GOAL = Goal(atoms=frozenset())

# The oracle's four dials in slot order, beside the bounds each has to fall inside.
_ORACLE_TOSS_PARAMS = (
    (ORACLE_THROW_STANDOFF, TOSS_DISTANCE_BOUNDS),
    (ORACLE_THROW_ROTATION, TOSS_ROTATION_BOUNDS),
    (ORACLE_RELEASE_SPEED_DEG_S, TOSS_SPEED_BOUNDS),
    (ORACLE_GRIPPER_RELEASE_MS, TOSS_RELEASE_MS_BOUNDS),
)

# The symbolic state in which the oracle should throw. **An atom set, not a gripper
# value.** The oracle branches on upstream's `Holding`, which adds a forward-kinematics
# conjunct a flat `core.State` cannot evaluate, so "the robot is holding the cube" is
# stated rather than implied by a feature that used to imply it. That is a clarification
# as much as a consequence: these tests are about the oracle's *branch*, and always
# were -- the classifiers' own semantics are `test_kb_predicate_parity.py`'s subject. The
# lifted `cube_z` rides along so the flat state still describes the same situation.
_HOLDING = {"atoms": HOLDING_ATOMS, "cube_z": 0.4}


def _act(*, atoms=INITIAL_ATOMS, **kwargs):
    """The oracle's choice for one symbolic situation."""
    return SkillOraclePolicy.get_labeled_action(
        state=state(abstract_atoms=atoms, **kwargs), env=_ENV, goal=_EMPTY_GOAL
    )


def test_the_oracle_picks_when_the_cube_is_on_the_ground() -> None:
    action = _act()
    assert action.action[0] == pytest.approx(Tossing3DEnvironment.pick_cube_id)


def test_the_oracle_passes_the_pick_no_parameters_at_all() -> None:
    """`pick_cube` derives its standoff and grasp rotation internally. The oracle used to
    supply a `(distance, rotation)` pair here; every slot must now be zero, so nothing
    downstream can read a stale dial out of one."""
    assert list(_act().action[1:]) == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_the_oracle_drives_and_throws_once_it_is_holding_the_cube() -> None:
    action = _act(**_HOLDING)
    assert action.action[0] == pytest.approx(Tossing3DEnvironment.move_to_toss_location_and_toss_id)
    assert list(action.action[1:]) == pytest.approx([
        ORACLE_THROW_STANDOFF,
        ORACLE_THROW_ROTATION,
        ORACLE_RELEASE_SPEED_DEG_S,
        ORACLE_GRIPPER_RELEASE_MS,
    ])


def test_the_oracle_solves_the_domain_in_exactly_two_skills_in_this_order() -> None:
    """The whole plan shape, walked symbolically: pick, then drive-and-throw. It was
    three, and a third rung reappearing would mean the oracle had drifted from the
    controllers `kinder_backend.py` can drive."""
    ids = [_act().action[0], _act(**_HOLDING).action[0]]
    assert ids == pytest.approx([
        Tossing3DEnvironment.pick_cube_id,
        Tossing3DEnvironment.move_to_toss_location_and_toss_id,
    ])


def test_where_the_robot_stands_does_not_change_what_the_oracle_does() -> None:
    """The branch is on `Holding` alone. Standing at the old throw standoff used to be
    what selected `Toss` over `MoveToThrowPose`; the composed controller drives itself, so
    the base pose must not reach the decision at all."""
    from .observations import BIN_X

    near_the_bin = _act(**_HOLDING, base_x=BIN_X - ORACLE_THROW_STANDOFF).action
    across_the_room = _act(**_HOLDING, base_x=0.0).action
    assert list(near_the_bin) == pytest.approx(list(across_the_room))


def test_the_oracle_offers_no_recovery_after_a_missed_toss() -> None:
    """It falls back to `PickCube`, whose grasp will fail to plan, rather than to some
    invented retrieval skill. There is nothing this domain can do once the cube is past
    the barrier, and pretending otherwise would hide the irreversibility it exists to
    exhibit."""
    action = _act(cube_x=2.6, cube_z=0.025, gripper=0.0)
    assert action.action[0] == pytest.approx(Tossing3DEnvironment.pick_cube_id)


@pytest.mark.parametrize(("value", "bounds"), _ORACLE_TOSS_PARAMS)
def test_every_oracle_toss_parameter_lies_inside_the_samplers_own_range(
    *, value: float, bounds: tuple[float, float]
) -> None:
    """An oracle drawing from outside the range a learner samples would be measuring a
    different skill from the one being learned -- and all four ranges narrowed in the
    migration, since they are now upstream's own rather than this package's. `1.35` in
    particular used to sit inside a `(1.10, 1.75)` standoff range and now sits inside
    `(1.25, 1.45)`."""
    assert bounds[0] <= value <= bounds[1]


def test_the_throw_standoff_is_upstreams_own_test_value() -> None:
    assert ORACLE_THROW_STANDOFF == 1.35


def test_the_throw_rotation_is_head_on() -> None:
    """Upstream's own value in `test_pick_ground_toss`, and the centre of
    `TOSS_ROTATION_BOUNDS` -- which is only about 0.8 degrees wide either way, so any
    other value would be a rounding of this one rather than a choice."""
    assert ORACLE_THROW_ROTATION == 0.0
    assert pytest.approx(sum(TOSS_ROTATION_BOUNDS) / 2) == ORACLE_THROW_ROTATION


def test_the_oracle_release_speed_is_upstreams_own_shipped_default() -> None:
    """140 deg/s is not a tuned number. It is the literal that was inline in
    `TossController.reset` before kb#8 made it a parameter, it is what
    `toss_profile_limits()` still returns by default, and it is the speed every committed
    Tossing3D number -- including the `10/10` at standoff 1.35 -- was measured at.

    It is also exactly the top of `TOSS_SPEED_BOUNDS`, which is what makes the oracle's
    throw the fastest one a learner could ever draw rather than an interior point it
    would have to find. If this number ever has to move, that is a new measurement, not a
    tweak.
    """
    assert ORACLE_RELEASE_SPEED_DEG_S == 140.0
    assert TOSS_SPEED_BOUNDS[1] == ORACLE_RELEASE_SPEED_DEG_S


def _measured_solving_band_ms() -> tuple[list[float], dict[float, int]]:
    """Release milliseconds that solve `5/5` at the oracle's speed, read out of
    PR #240's committed grid rather than retyped.

    > **Measured under the three-skill decomposition, and left as published.** That grid
    > drove `Pick` -> `MoveToThrowPose(1.35)` -> `Toss(speed, ms)` as three separate
    > controllers. `MoveToTossLocationAndToss` plans the base motion, the windup and the
    > swing together, so the swing starts from an arm configuration the planner chose
    > rather than from a separately commanded windup. The tests below are therefore
    > **provenance** checks -- they say where the shipped `792` came from and that the
    > committed grid still says it -- and are not evidence about the composed controller.
    > `test_kinder_fidelity.py` is what measures the composed controller.
    """
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
    speed; upstream rounds the millisecond to a whole one, so the midpoint is 792. See
    `_measured_solving_band_ms`' own note for why this is provenance and not evidence
    about the composed controller."""
    band, _ = _measured_solving_band_ms()
    assert len(band) == 2, "the 5/5 band at 140 deg/s is two grid cells wide"
    assert round((band[0] + band[-1]) / 2.0) == ORACLE_GRIPPER_RELEASE_MS
    assert band[0] < ORACLE_GRIPPER_RELEASE_MS < band[-1]


def test_the_oracle_release_ms_clears_the_nearest_measured_failure_by_80ms() -> None:
    """The grid steps 57.9 ms: this is the measured margin, the real one >= 28.8 ms."""
    band, solved_by_ms = _measured_solving_band_ms()
    below = max(ms for ms in solved_by_ms if ms < band[0] and solved_by_ms[ms] < 5)
    above = min(ms for ms in solved_by_ms if ms > band[-1] and solved_by_ms[ms] < 5)
    assert ORACLE_GRIPPER_RELEASE_MS - below >= 80.0
    assert above - ORACLE_GRIPPER_RELEASE_MS >= 80.0


def test_the_label_names_the_skill_and_its_objects() -> None:
    """`LabeledAction.label` is what the renderer burns into the frame, so it has to say
    what actually happened rather than just which skill ran. The pick carries no
    `params=` suffix because it has no parameters -- an empty `params=[]` would invite a
    reader to look for a dial that does not exist."""
    label = _act().label
    assert label == "PickCube(robot, cube_0, cuboid_barrier)"


def test_the_toss_label_carries_all_four_dials_in_upstreams_object_order() -> None:
    """A clip of a throw has to say how hard and from where, or two clips at different
    parameters are indistinguishable. The object order is upstream's own
    `(robot, target, held, barrier)`, so the label reads as the controller call it is."""
    label = _act(**_HOLDING).label
    assert label.startswith("MoveToTossLocationAndToss(robot, bin_0, cube_0, cuboid_barrier)")
    assert "params=[1.35, 0.0, 140.0, 792.0]" in label


def test_the_provider_forwards_its_configured_standoff() -> None:
    """The standoff is a constructor field rather than a constant read off the policy
    module, so the CLI can say which one to throw from. Probed at a value inside
    `TOSS_DISTANCE_BOUNDS` but away from the default: forwarding is the property, and a
    probe outside upstream's own sampling range would be asserting that the provider
    passes through a standoff the controller was never measured at."""
    configured = 1.28
    assert TOSS_DISTANCE_BOUNDS[0] < configured < TOSS_DISTANCE_BOUNDS[1]
    assert configured != ORACLE_THROW_STANDOFF

    oracle = Tossing3DOracle(env=_ENV, throw_standoff=configured)
    action = oracle.get_labeled_action(
        state=state(abstract_atoms=HOLDING_ATOMS, cube_z=0.4), goal=_EMPTY_GOAL
    )
    assert action.action[1] == pytest.approx(configured)
