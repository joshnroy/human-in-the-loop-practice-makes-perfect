"""A pick that ends up against the barrier backs out, and one that starts there escapes.

t3-relaunch-20261023 EES seed 2, PickCube seq 68: the descent beside the barrier touched
it and pushed against it for the rest of its 400 steps, leaving the arm in contact.
Every later pick then refused ("Robot starts in collision with ...cuboid_barrier..."),
#572 marked the pick blocked, and the only human skill moves the cube, not the arm, so
EES reset until `repeated_no_robot_action` ended the run (215 refusals). Seed 0 ended the
same way (212). The two states are frozen here from that run's recording: the last tick
before seq 68, and the last tick of it.

Upstream kinder-baselines now (a) aborts a pick at its first unexpected contact and
retraces the executed path out, (b) retraces the same way when the step budget runs
out (`KinderBackend.run_controller` calls `on_step_limit`), and (c) escapes a start in
contact along a path whose clearance never decreases.
"""

import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import PICK_PLANNABLE, PICKUP_UNBLOCKED

pytestmark = [
    pytest.mark.xdist_group("test_barrier_stall_recovery"),
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KINDER"),
]

_FIXTURES = Path(__file__).parent / "fixtures" / "pick_plannable"
_PICK_START = "t3relaunch_ees_s2_seq68_barrier_pick_start.json"
_STALL_END = "t3relaunch_ees_s2_seq68_barrier_stall_end.json"
_BARRIER = "collider:cuboid_barrier:9"


@pytest.fixture(scope="module")
def live_env() -> Iterator[Tossing3DEnvironment]:
    env = Tossing3DEnvironment()
    env.reset_to_seed(seed=125)
    yield env
    env.close()


def _restore(*, env: Tossing3DEnvironment, fixture: str) -> Any:
    plain = json.loads((_FIXTURES / fixture).read_text())
    return env.restore_plain_snapshot(plain=plain, dtype=np.float64)


def _touching(*, env: Tossing3DEnvironment) -> str | None:
    """What a fresh pick finds the robot touching where it is now."""
    backend = env.backend()
    state = backend.live_state()
    sim = backend._pick_collision_sim(state=state)  # noqa: SLF001
    try:
        controller = backend._ground_controller(  # noqa: SLF001
            module="tossing",
            key="pick_cube",
            object_names=(backend.robot_name, backend.cube_name, backend.barrier_name),
            state=state,
            pybullet_sim=sim,
        )
        return controller.current_configuration_collision(state)
    finally:
        sim.close()


def _pick(*, env: Tossing3DEnvironment) -> Any:
    return env.take_action(action=np.array([float(env.pick_cube_id), 0.0, 0.0, 0.0, 0.0]))


def test_the_frozen_states_are_the_stall(*, live_env: Tossing3DEnvironment) -> None:
    """Clear before the pick, against the barrier after it."""
    _restore(env=live_env, fixture=_PICK_START)
    assert _touching(env=live_env) is None
    _restore(env=live_env, fixture=_STALL_END)
    assert _touching(env=live_env) == _BARRIER


def test_the_stalling_pick_aborts_at_the_barrier_and_backs_out(
    *, live_env: Tossing3DEnvironment
) -> None:
    """The recorded pick now stops at its first contact with the barrier, well inside
    its 400 steps, retraces out, and reports a failure; the arm ends clear and the next
    pick plans, so the pick is not blocked."""
    env = live_env
    _restore(env=env, fixture=_PICK_START)
    after = _pick(env=env)
    error = env.last_skill_error() or ""
    assert "PickCube aborted: unexpected contact with" in error
    assert "cuboid_barrier" in error
    assert sum(env.last_controller_steps()) < env.backend().pick_step_limit
    assert _touching(env=env) is None
    assert env.backend().pick_cube_plan_failure() is None
    assert PICKUP_UNBLOCKED.holds(after, (env.cube,))


def test_a_pick_from_the_stalled_state_escapes_instead_of_refusing(
    *, live_env: Tossing3DEnvironment
) -> None:
    """From the arm left on the barrier the pick is plannable again (it was refused
    with "Robot starts in collision"), and dispatched it ends clear of everything."""
    env = live_env
    stalled = _restore(env=env, fixture=_STALL_END)
    assert env.backend().pick_cube_plan_failure() is None
    assert PICK_PLANNABLE.holds(stalled, (env.robot, env.cube))
    _pick(env=env)
    assert sum(env.last_controller_steps()) > 0
    assert _touching(env=env) is None
