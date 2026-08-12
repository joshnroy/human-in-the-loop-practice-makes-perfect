"""Offline tests for `Tossing3DTasks`, and specifically for its reset-free path.

Everything here runs without MuJoCo. `Tossing3DEnvironment.reset_to_seed` is the one
door to the simulator that task sampling goes through, so a test can pin "no simulator
operation happened" by replacing that method with one that raises -- no live scene, and
no dependence on the optional `tossing3d` extra CI never installs.
"""

import pytest

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import IN_BIN
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks

from .observations import state


class _NoSimulatorEnvironment(Tossing3DEnvironment):
    """A `Tossing3DEnvironment` whose only simulator entry point is a tripwire.

    Deliberately `reset_to_seed` rather than the backend: it is the single method
    `hard_reset`, `set_state` and `build_task` all funnel through (see its docstring),
    so trapping it catches every way this domain can rebuild a scene."""

    def reset_to_seed(self, *, seed: int) -> None:  # type: ignore[override]
        raise AssertionError(f"reset_to_seed(seed={seed}) was called, so the simulator was rebuilt")


def _env_sitting_at_a_scene() -> _NoSimulatorEnvironment:
    """An environment already inhabiting a scene, as `hard_reset` would have left it."""
    env = _NoSimulatorEnvironment()
    env.current_state = state(env=env, cube_x=0.9, cube_y=0.1, seed=125)
    return env


def test_sampling_a_train_task_in_place_performs_no_simulator_operation() -> None:
    """The defect this file exists to close. Under `practice_reset_policy=never` the
    practice loop declines to install the sampled task's initial state -- but on this
    domain `sample_train_task` had already rebuilt the MuJoCo scene to produce one, so
    the arm was reset every cycle while `num_practice_resets` correctly reported 0 for
    the branch it counts. Asserted against the simulator, not against that counter."""
    env = _env_sitting_at_a_scene()
    tasks = Tossing3DTasks(env=env, seed=0)

    task = tasks.sample_train_task_in_place()

    assert task.initial_state is env.get_current_state()


def test_sampling_a_train_task_in_place_leaves_the_scene_seed_stream_untouched() -> None:
    """A reset-free run inhabits one scene for its whole length, so there is no scene
    to draw. Consuming a seed anyway would make the two arms' train streams diverge for
    a reason unrelated to the manipulation."""
    env = _env_sitting_at_a_scene()
    tasks = Tossing3DTasks(env=env, seed=0)
    before = tasks.train_rng.bit_generator.state

    tasks.sample_train_task_in_place()

    assert tasks.train_rng.bit_generator.state == before


def test_the_in_place_task_asks_for_the_same_goal_as_a_sampled_one() -> None:
    """What makes the fix sound: `Predicate.__call__` discards the state it is handed
    (`core/problem/tasks/types.py`, `Predicate.__call__`), and this domain has one goal
    family over `ClassVar` objects -- so the goal is scene-independent and a `Task`
    needs a `State` only to satisfy the type. If either of those stopped holding, an
    in-place task would be asking for something other than what a sampled one asks
    for, and the two arms would no longer be comparable."""
    env = _env_sitting_at_a_scene()
    tasks = Tossing3DTasks(env=env, seed=0)

    goal = tasks.sample_train_task_in_place().goal

    assert goal.atoms == frozenset({
        IN_BIN(state=env.get_current_state(), objects=(env.cube, env.bin))
    })


def test_sampling_a_train_task_the_ordinary_way_still_rebuilds_the_scene() -> None:
    """The scheduled arm is unchanged, and that is checked rather than assumed: it is
    the arm every committed Tossing3D number was measured on."""
    env = _env_sitting_at_a_scene()
    tasks = Tossing3DTasks(env=env, seed=0)

    with pytest.raises(AssertionError, match="reset_to_seed"):
        tasks.sample_train_task()
