"""The tests that genuinely drive KINDER.

This domain integrates KINDER rather than reimplementing it, so these tests pin the
*integration boundary* -- that what this repo's adapter reads, dispatches and asserts
agrees with what upstream actually produces. They deliberately do NOT re-test anything
KINDER already guarantees: there is no reimplementation of its physics or its goal check
to check against. `test_in_goal_region_agrees_with_kinders_own_goal_check` is the shape
of the whole file -- run the real simulator, ask upstream's `_check_goals()`, and require
our predicate to return the same verdict.

Two of them are a different kind, and are labelled as such below:
`test_goal_region_bounds_match_the_variants_task_json` is a version tripwire against a
literal from KINDER's task JSON, and `test_a_full_power_toss_overshoots_the_goal_region`
characterises upstream's physics. Both are only meaningful because `pyproject.toml` pins
KINDER to an exact commit -- they are what makes an upstream bump that moves the goal
region or the swing dynamics fail loudly instead of silently restating every number
measured here.

Everything else in this directory checks the adapter's arithmetic against itself, which
needs no simulator. These are skipped where `kindergarden` is not installed (CI
included) rather than deleted, because the alternative is asserting nothing about the
seam anywhere. Run them with the optional dependency installed; see
`src/hitl_pmp/environments/tossing3d/README.md`.
"""

import resource

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import IN_GOAL_REGION, REACHABLE
from hitl_pmp.environments.tossing3d.skill_oracle_policy import SkillOraclePolicy

from .conftest import GOAL_REGION, kinder_available

_ENV = Tossing3DEnvironment

pytestmark = kinder_available

_SHARED: list[Tossing3DEnvironment] = []


def shared_env() -> Tossing3DEnvironment:
    """One simulator for the whole module -- opening it compiles a MuJoCo model and
    connects a PyBullet client, so it is built once and reused. A module-level cache
    rather than a pytest fixture: this project's lint bans positional parameters
    outright (ruff PLR0917, max-positional-args = 0), and a fixture argument is one."""
    if not _SHARED:
        _SHARED.append(Tossing3DEnvironment())
    return _SHARED[0]


def _act(*, env: Tossing3DEnvironment, skill: int, param0: float = 0.0, param1: float = 0.0):
    return env.take_action(action=np.array([float(skill), float(param0), float(param1)]))


def _oracle_solve(*, env: Tossing3DEnvironment, seed: int, swing: float):
    state = env.reset_to_seed(seed=seed)
    for skill, param in (
        (_ENV.SKILL_PICK, SkillOraclePolicy.ORACLE_PICK_DISTANCE),
        (_ENV.SKILL_MOVE_TO_THROW_POSE, 0.0),
        (_ENV.SKILL_TOSS, swing),
    ):
        state = _act(env=env, skill=skill, param0=param)
    return state


def test_goal_region_bounds_match_the_variants_task_json() -> None:
    """A version tripwire, not a differential test: `GOAL_REGION` is a literal copy of
    the pinned KINDER's task JSON, so this fires if an upstream bump moves the region
    out from under every number measured against it."""
    assert shared_env().goal_region_bounds() == pytest.approx(GOAL_REGION)


def test_in_goal_region_agrees_with_kinders_own_goal_check() -> None:
    """The fidelity property this domain's headline number rests on: this domain's
    `InGoalRegion` predicate must be KINDER's `_check_goals`, not a lookalike.

    Checked by driving the real simulator (a random walk of skills, which lands the cube
    anywhere from its start pose to inside the bin) and comparing the two verdicts on
    every state visited -- rather than by asserting on synthetic positions, which would
    only test this file's own arithmetic.
    """
    env = shared_env()
    backend = env.backend()
    rng = np.random.default_rng(0)
    compared = 0
    for seed in (0, 1, 2):
        env.reset_to_seed(seed=seed)
        for _ in range(4):
            skill = int(
                rng.choice([_ENV.SKILL_PICK, _ENV.SKILL_MOVE_TO_THROW_POSE, _ENV.SKILL_TOSS])
            )
            state = _act(env=env, skill=skill, param0=float(rng.uniform(0.25, 1.25)))
            ours = IN_GOAL_REGION.holds(state, (_ENV.cube, _ENV.goal_region))
            assert ours == backend.check_goals(), (
                f"seed {seed}: InGoalRegion said {ours}, KINDER's _check_goals disagreed"
            )
            compared += 1
    assert compared == 12


def test_reset_to_the_same_seed_reproduces_the_same_initial_state() -> None:
    """`set_state` reinstalls a State by re-running its KINDER seed, which is only sound
    if a reset is deterministic. Checked after driving the simulator far away in
    between, which is the case the harness actually creates."""
    env = shared_env()
    first = env.reset_to_seed(seed=11)
    cube_before = np.array(first[_ENV.cube])
    robot_before = np.array(first[_ENV.robot])
    _oracle_solve(env=env, seed=11, swing=1.0)
    again = env.reset_to_seed(seed=11)
    assert np.array_equal(np.array(again[_ENV.cube]), cube_before)
    assert np.array_equal(np.array(again[_ENV.robot]), robot_before)


def test_the_oracle_swing_actually_reaches_the_goal_region() -> None:
    """The oracle's swing constant is the achievable ceiling every learning number here
    is read against, so it has to be a measured fact rather than a comment."""
    env = shared_env()
    solved = 0
    for seed in (0, 1, 2):
        state = _oracle_solve(env=env, seed=seed, swing=SkillOraclePolicy.ORACLE_SWING)
        solved += int(IN_GOAL_REGION.holds(state, (_ENV.cube, _ENV.goal_region)))
    assert solved >= 2, f"oracle solved only {solved}/3 -- ORACLE_SWING may need remeasuring"


def test_a_full_power_toss_overshoots_the_goal_region() -> None:
    """A characterisation of upstream's physics rather than of this adapter, and sound
    only because `pyproject.toml` pins KINDER exactly: swing=1.0 is KINDER's own demo
    toss and lands the cube in the bin at x ~ 2.22, past the region's 2.10 edge. This is
    what makes the swing dial worth learning: the obvious value is the wrong one, and a
    KINDER bump that changed it would invalidate the swing table this domain's README
    reports.

    Asserted on seeds 0 and 2, not seed 1. On seed 1 the grasp is marginal and the cube
    slips out of the gripper during `move_to_target`, landing at x ~ 1.58 without ever
    being tossed -- so seed 1 measures the Pick, not the swing. That is the same seed
    `test_the_oracle_swing_actually_reaches_the_goal_region` tolerates by asserting 2 of
    3 rather than 3 of 3.
    """
    for seed in (0, 2):
        state = _oracle_solve(env=shared_env(), seed=seed, swing=1.0)
        assert state.get(obj=_ENV.cube, feature_name="x") > GOAL_REGION[3], (
            f"seed {seed}: full-power toss did not clear the region's far edge"
        )
        assert not IN_GOAL_REGION.holds(state, (_ENV.cube, _ENV.goal_region))


def test_a_tossed_cube_is_unreachable() -> None:
    """The domain's whole reason for existing: after a toss the cube is past the
    barrier, so no further Pick can ever apply."""
    env = shared_env()
    assert REACHABLE.holds(env.reset_to_seed(seed=2), (_ENV.cube, _ENV.barrier))
    state = _oracle_solve(env=env, seed=2, swing=SkillOraclePolicy.ORACLE_SWING)
    assert not REACHABLE.holds(state, (_ENV.cube, _ENV.barrier))


def test_a_kinder_planning_failure_is_a_no_op_not_a_crash() -> None:
    """Inverse kinematics genuinely has no solution for some base poses. That is a
    failure of the Pick skill, and `take_action` has to stay total over the Box."""
    env = shared_env()
    env.reset_to_seed(seed=0)
    for rot in np.linspace(_ENV().pick_rot_low, _ENV().pick_rot_high, 5):
        assert _act(env=env, skill=_ENV.SKILL_PICK, param0=0.5, param1=float(rot)) is not None


def test_every_swing_the_prior_can_draw_stays_executable() -> None:
    """Guards the seam between the prior and the simulator: a swing the sampler can draw
    but the controllers reject would show up as an unexplained plateau, not an error."""
    env = shared_env()
    for swing in (env.swing_low, 0.5, env.swing_high):
        assert _oracle_solve(env=env, seed=0, swing=swing) is not None


def test_skill_executions_do_not_leak_memory() -> None:
    """`KinderBackend._release` has to reclaim the PyBullet client each grounded
    controller opens, or a sweep OOMs the machine rather than finishing.

    This is a regression test for a real incident, not a hypothetical: grounding a
    controller per skill execution leaked one live `p.connect(p.DIRECT)` plus the Kinova
    URDF and meshes -- ~150 MB per Pick, ~315 MB per Toss -- which took a 40-step run to
    18.7 GB and put a 60 GB box on an OOM trajectory.

    Peak RSS via `resource`, so no dependency beyond the stdlib, and the threshold sits
    with enormous margin on both sides: leaking costs >6 GB over these 40 executions,
    while reclaiming holds growth to roughly nothing.
    """
    env = shared_env()
    env.reset_to_seed(seed=0)
    # Warm up first: the one-off cost of the first grounding of each controller is not
    # the leak, and folding it into the measurement would only add noise.
    for skill in (_ENV.SKILL_PICK, _ENV.SKILL_MOVE_TO_THROW_POSE, _ENV.SKILL_TOSS):
        _act(env=env, skill=skill, param0=0.5)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for index in range(40):
        env.reset_to_seed(seed=index % 3)
        _act(env=env, skill=_ENV.SKILL_PICK, param0=0.55)
        _act(env=env, skill=_ENV.SKILL_TOSS, param0=0.75)
    # ru_maxrss is in kilobytes on Linux.
    grew_mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) / 1024
    assert grew_mb < 500, f"peak RSS grew {grew_mb:.0f} MB over 40 skill executions"
