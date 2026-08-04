"""The tests that genuinely drive KINDER.

These are the ones that can go stale silently: everything else in this directory checks
this port against itself, and only these check it against the benchmark. They are
skipped where `kindergarden` is not installed (CI included) rather than deleted,
because the alternative -- asserting nothing about fidelity anywhere -- is how a port
drifts from the thing it claims to reproduce. Run them with the optional dependency
installed; see `src/hitl_pmp/environments/tossing3d/README.md`.
"""

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
    assert shared_env().goal_region_bounds() == pytest.approx(GOAL_REGION)


def test_in_goal_region_agrees_with_kinders_own_goal_check() -> None:
    """The fidelity property this port's headline number rests on: this domain's
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
    """swing=1.0 is KINDER's own demo toss and lands the cube in the bin at x ~ 2.22,
    past the region's 2.10 edge. This is what makes the swing dial worth learning: the
    obvious value is the wrong one."""
    state = _oracle_solve(env=shared_env(), seed=1, swing=1.0)
    assert state.get(obj=_ENV.cube, feature_name="x") > GOAL_REGION[3]
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
