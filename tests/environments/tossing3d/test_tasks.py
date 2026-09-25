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


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("kinder") is None, reason="KINDER simulator dependency"
)
def test_evaluation_tasks_still_place_the_bin_on_the_far_side() -> None:
    """Restricting practice resets to the robot side leaves evaluation alone: a test
    task's full reset still places the bin beyond the barrier."""
    env = Tossing3DEnvironment()
    tasks = Tossing3DTasks(env=env, seed=0)
    try:
        for _ in range(3):
            task = tasks.sample_test_task()
            bin_x = task.initial_state.get(obj=env.bin, feature_name="x")
            barrier_x = task.initial_state.get(obj=env.barrier, feature_name="x")
            assert bin_x > barrier_x
    finally:
        env.close()


# --- practice scenes keep the bin on the robot side ---------------------------------

BLOCK = (-0.33, -1.9, 0.3, -1.3)
# The first three test tasks at seed 0, recorded before practice scenes changed. They
# must not move: every evaluation number is comparable only while these hold.
PINNED_TEST_BINS = (
    (1.8369, 1.4474, 357381689),
    (3.1035, 1.5562, 1109584189),
    (2.7486, -1.4003, 861111389),
)


def _bin_xy(*, env, state) -> tuple[float, float]:
    return (state.get(obj=env.bin, feature_name="x"), state.get(obj=env.bin, feature_name="y"))


def _assert_in_block_and_valid(*, env, state) -> None:
    from hitl_pmp.environments.tossing3d.bin_placement import BinPlacementRules
    from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend

    x, y = _bin_xy(env=env, state=state)
    assert BLOCK[0] - 1e-3 <= x <= BLOCK[2] + 1e-3
    assert BLOCK[1] - 1e-3 <= y <= BLOCK[3] + 1e-3
    geometry = KinderBackend.toss_feasibility_geometry(snapshot=env.backend().snapshot())
    robot = BinPlacementRules.aabb(
        center=geometry.robot_pose[:2], size=geometry.robot_size, yaw=geometry.robot_pose[2]
    )
    BinPlacementRules.check(
        bin_aabb=BinPlacementRules.aabb(center=(x, y), size=(0.3, 0.3), yaw=geometry.bin_pose[2]),
        robot_aabb=robot,
        cube_spawn=((0.5, -0.25, 0.75, 0.25),),
        context="practice scene",
    )


needs_kinder = pytest.mark.skipif(
    __import__("importlib").util.find_spec("kinder") is None, reason="KINDER simulator dependency"
)


@needs_kinder
def test_every_practice_scene_puts_the_bin_in_the_robot_side_block() -> None:
    """Initial (`hard_reset`, what `never` practices in), sampled train tasks, and the
    rebuild `reset_to_task` does each `scheduled` period all land in the block."""
    from hitl_pmp.environments.tossing3d.sides import Tossing3DSide

    env = Tossing3DEnvironment(scene_bin_destination=Tossing3DSide.ROBOT)
    tasks = Tossing3DTasks(env=env, seed=0)
    try:
        env.hard_reset()
        _assert_in_block_and_valid(env=env, state=env.get_current_state())
        for _ in range(3):
            task = tasks.sample_train_task()
            _assert_in_block_and_valid(env=env, state=task.initial_state)
            env.set_state(state=task.initial_state)
            rebuilt = env.get_current_state()
            _assert_in_block_and_valid(env=env, state=rebuilt)
            assert _bin_xy(env=env, state=rebuilt) == pytest.approx(
                _bin_xy(env=env, state=task.initial_state), abs=1e-6
            )
    finally:
        env.close()


@needs_kinder
def test_evaluation_test_tasks_are_unchanged() -> None:
    env = Tossing3DEnvironment()
    tasks = Tossing3DTasks(env=env, seed=0)
    try:
        for bx, by, seed in PINNED_TEST_BINS:
            initial = tasks.sample_test_task().initial_state
            assert int(initial.get(obj=env.scene, feature_name="seed")) == seed
            assert _bin_xy(env=env, state=initial) == pytest.approx((bx, by), abs=1e-3)
    finally:
        env.close()
