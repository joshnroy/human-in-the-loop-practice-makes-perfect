"""`Tossing3DTasks`, and specifically its reset-free path.

**These used to run offline and now need a simulator, for a reason worth stating.**
`goal_atom()` resolves `MovableInGoalRegion` out of the live abstraction, and
`Tossing3DEnvironment.abstraction()` builds the scene if there is not one -- so the goal
this domain asks for cannot be constructed without MuJoCo any more. The old
`_NoSimulatorEnvironment`, which trapped `reset_to_seed` and never built a scene at all,
would now trip on its own goal.

The property being protected is unchanged, and it is the one that matters: **sampling a
task in place must rebuild nothing.** So the tripwire is armed *after* the scene exists
rather than instead of it, which tests the same thing on the states the domain can
actually reach.
"""

import pytest

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import MOVABLE_IN_GOAL_REGION, Tossing3DPredicates
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks

from .conftest import requires_kinder

pytestmark = requires_kinder


def _arm_the_tripwire(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any further scene rebuild fail loudly.

    `reset_to_seed` rather than the backend: it is the single method `hard_reset`,
    `set_state` and `build_task` all funnel through (see its docstring), so trapping it
    catches every way this domain can rebuild a scene.

    Patched on the class, not the instance: `Tossing3DEnvironment` is a pydantic model and
    refuses an attribute that is not a declared field. `monkeypatch` undoes it at teardown,
    so the shared scene is unaffected once the test ends.
    """

    # Positional `self` because it stands in for a bound method on the environment.
    def explode(self: Tossing3DEnvironment, *, seed: int) -> None:  # noqa: PLR0917
        del self
        raise AssertionError(f"reset_to_seed(seed={seed}) was called, so the scene was rebuilt")

    monkeypatch.setattr(Tossing3DEnvironment, "reset_to_seed", explode)


def test_sampling_a_train_task_in_place_performs_no_simulator_operation(
    *, live_env: Tossing3DEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect this file exists to close. Under `practice_reset_policy=never` the
    practice loop declines to install the sampled task's initial state -- but on this
    domain `sample_train_task` had already rebuilt the MuJoCo scene to produce one, so
    the arm was reset every cycle while `num_practice_resets` correctly reported 0 for
    the branch it counts. Asserted against the simulator, not against that counter."""
    tasks = Tossing3DTasks(env=live_env, seed=0)
    _arm_the_tripwire(monkeypatch=monkeypatch)

    task = tasks.sample_train_task_in_place()

    assert task.initial_state is live_env.get_current_state()


def test_sampling_a_train_task_in_place_leaves_the_scene_seed_stream_untouched(
    *, live_env: Tossing3DEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reset-free run inhabits one scene for its whole length, so there is no scene
    to draw. Consuming a seed anyway would make the two arms' train streams diverge for
    a reason unrelated to the manipulation."""
    tasks = Tossing3DTasks(env=live_env, seed=0)
    _arm_the_tripwire(monkeypatch=monkeypatch)
    before = tasks.train_rng.bit_generator.state

    tasks.sample_train_task_in_place()

    assert tasks.train_rng.bit_generator.state == before


def test_the_in_place_task_asks_for_the_same_goal_as_a_sampled_one(
    *, live_env: Tossing3DEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What makes the fix sound: `Predicate.__call__` discards the state it is handed
    (`core/problem/tasks/types.py`, `Predicate.__call__`), and this domain has one goal
    family over `ClassVar` objects -- so the goal is scene-independent and a `Task`
    needs a `State` only to satisfy the type. If either of those stopped holding, an
    in-place task would be asking for something other than what a sampled one asks
    for, and the two arms would no longer be comparable."""
    tasks = Tossing3DTasks(env=live_env, seed=0)
    expected = Tossing3DPredicates.get(
        abstraction=live_env.abstraction(), name=MOVABLE_IN_GOAL_REGION
    )(state=live_env.get_current_state(), objects=(live_env.cube,))
    _arm_the_tripwire(monkeypatch=monkeypatch)

    goal = tasks.sample_train_task_in_place().goal

    assert goal.atoms == frozenset({expected})


def test_the_goal_names_only_the_cube_and_no_longer_the_bin(
    *, live_env: Tossing3DEnvironment
) -> None:
    """**Upstream's goal is unary, and the change is not cosmetic.** It used to be
    `InBin(cube_0, bin_0)`, which asserted containment in a box carried on the bin object
    under a stated assumption that the bin's interior *is* the scored region. Upstream
    scores `["on", "cube_0", "blocks_goal_region"]` -- a ground region the bin merely sits
    near -- so naming the bin in the goal made this domain ask for something adjacent to,
    rather than identical with, what KINDER actually checks."""
    tasks = Tossing3DTasks(env=live_env, seed=0)

    atom = tasks.goal_atom()

    assert atom.objects == (live_env.cube,)
    assert atom.predicate.name == MOVABLE_IN_GOAL_REGION


def test_sampling_a_train_task_the_ordinary_way_still_rebuilds_the_scene(
    *, live_env: Tossing3DEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scheduled arm is unchanged, and that is checked rather than assumed: it is
    the arm every committed Tossing3D number was measured on."""
    tasks = Tossing3DTasks(env=live_env, seed=0)
    _arm_the_tripwire(monkeypatch=monkeypatch)

    with pytest.raises(AssertionError, match="reset_to_seed"):
        tasks.sample_train_task()
