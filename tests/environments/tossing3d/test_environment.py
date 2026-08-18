"""`Tossing3DEnvironment`: the KINDER -> `core.State` translation and the action encoding.

**This file used to be entirely offline and is now mostly simulator-backed.** The reason
is not a change of testing taste: the boundary type the offline half was built on
(`KinderObservation`, a dict of plain floats) no longer exists, because the predicates are
upstream's now and a predicate that runs forward kinematics cannot be evaluated against a
dict. See `conftest.py` for the full trade.

What is still genuinely offline is kept offline and marked as such, because it is the part
that protects the laziness itself -- if constructing a `Tossing3DEnvironment` ever imported
MuJoCo, `hitl_pmp.cli` would stop importing on a machine without it, and a test that needed
a simulator to notice could not say so.
"""

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend

from .conftest import CANONICAL_SEED, requires_kinder

# --- genuinely offline: no simulator is built, and that is the property ----------------


def test_constructing_the_environment_imports_no_simulator(*, no_kinder_import) -> None:
    """The whole reason `KinderBackend` is lazy. `hitl_pmp.cli` imports every registered
    environment's CLI, so if constructing one of these reached MuJoCo then `--env
    lightswitch` would stop working on a machine without the optional extra."""
    env = Tossing3DEnvironment()
    assert env.variant == "o1"
    del no_kinder_import  # the fixture is the assertion; see conftest


def test_get_valid_actions_is_empty_because_the_parameters_are_continuous() -> None:
    assert Tossing3DEnvironment().get_valid_actions() == []


def test_the_action_space_has_one_id_slot_and_four_parameter_slots() -> None:
    """Four, not two. `move_to_toss_location_and_toss` is the widest skill -- distance,
    rotation, tossing speed, release millisecond -- and it is one skill rather than the
    two it used to be, so its four parameters have to fit in a single action vector."""
    assert Tossing3DEnvironment.action_space.shape == (5,)


def test_the_backend_overrides_no_task_config_so_the_scene_is_upstreams() -> None:
    """**The retired choice, pinned as an absence.** This domain used to select between
    upstream's `Tossing3D-o1.json` and a copy of it committed here; passing no
    `task_config_path` is what "run whatever the installed KINDER ships" means at the
    seam, and a future change that reintroduced an override would silently reintroduce
    the fork this removed. Reads the backend rather than the simulator, so it stays
    offline."""
    backend = Tossing3DEnvironment().backend()
    assert not hasattr(backend, "task_config_path")
    assert backend.env_id == "kinder/Tossing3D-o1-v0"


def test_a_variant_this_domains_symbolic_layer_cannot_describe_is_refused() -> None:
    """`o2` needs two cubes in the goal region and the symbolic layer here is single-cube,
    so a run labelled `o2` would be measuring something this package cannot describe.
    Raised rather than silently no-oped, and raised from `backend()` so it costs a
    construction rather than a simulator build."""
    env = Tossing3DEnvironment(variant="o2")
    with pytest.raises(ValueError, match="o1 scene"):
        env.backend()


def test_the_noop_id_is_not_a_real_skill_id() -> None:
    """Negative rather than 2, so that adding a third controller can never silently turn
    every no-op into it."""
    env = Tossing3DEnvironment()
    assert env.noop_id not in {env.pick_cube_id, env.move_to_toss_location_and_toss_id}


def test_a_non_finite_action_is_recorded_as_a_no_op_rather_than_raising() -> None:
    """`take_action` must be total over its Box action space, and `round(inf)` raises.
    Still offline: the finiteness check runs before anything touches the scene."""
    env = Tossing3DEnvironment()
    assert env._execute(action=np.array([np.inf, 0.0, 0.0, 0.0, 0.0])) == []  # noqa: SLF001
    assert env.last_skill_error() is not None
    assert "non-finite" in env.last_skill_error()


# --- simulator-backed from here on ----------------------------------------------------
#
# Marked per test rather than with a module-level `pytestmark`, because the offline tests
# above must keep running on a checkout with an empty `reference/` -- they are the ones
# that prove no simulator is needed.


@requires_kinder
def test_the_declared_type_schemas_are_kinders_own(*, live_env: Tossing3DEnvironment) -> None:
    """**The check `environment.py`'s own docstring promises, which did not exist.**

    The two `core.Type`s are written out as literals rather than derived at import, so
    that this class stays importable without MuJoCo. That is the right trade, but it makes
    the schemas a *copy* -- and a copy has no way of noticing that the original moved. A
    rename or a reordering upstream would leave the translation reading the right feature
    names off the wrong indices, which is silent and produces plausible numbers.

    Compared against `MujocoObjectTypeFeatures`, upstream's own declaration, element for
    element and in order: the translation is positional, so order is as load-bearing as
    membership.
    """
    from kinder.envs.dynamic3d.object_types import (
        MujocoMovableObjectType,
        MujocoObjectTypeFeatures,
        MujocoTidyBotRobotObjectType,
    )

    del live_env  # only here so a broken KINDER install fails once, in the fixture
    assert tuple(MujocoObjectTypeFeatures[MujocoTidyBotRobotObjectType]) == (
        Tossing3DEnvironment.robot_type.feature_names
    )
    assert tuple(MujocoObjectTypeFeatures[MujocoMovableObjectType]) == (
        Tossing3DEnvironment.movable_type.feature_names
    )


@requires_kinder
def test_translation_carries_every_feature_kinder_reports_for_every_object(
    *, live_env: Tossing3DEnvironment
) -> None:
    """The translation is lossless, and that is what lets the abstractor be handed a state
    back. A subset would be cheaper and is exactly what this domain used to do -- four of
    the robot's thirty-eight features -- but `_check_holding` runs forward kinematics off
    the arm joints, so a lossy state is one the abstractor rejects or, worse, misreads."""
    state = live_env.get_current_state()
    kinder_state = live_env.backend().kinder_state()

    for obj in (live_env.robot, live_env.cube, live_env.bin, live_env.barrier):
        kinder_object = kinder_state.get_object_from_name(obj.name)
        for feature_name in obj.type.feature_names:
            assert state.get(obj=obj, feature_name=feature_name) == pytest.approx(
                float(kinder_state.get(kinder_object, feature_name))
            ), f"{obj.name}.{feature_name} did not survive the translation"


@requires_kinder
def test_the_bin_no_longer_carries_a_scored_region_box(*, live_env: Tossing3DEnvironment) -> None:
    """**A deletion pinned as an absence, because reintroducing it would be a regression
    that looks like a feature.** The bin used to carry six extra features (`x_min` ..
    `z_max`) so a hand-written `InBin` could be a pure function of the `State`. That box
    was a second copy of a region upstream owns, and it shipped wrong once already -- the
    *uninflated* range, 2/3 of the true width on the one axis a toss controls.

    `MovableInGoalRegion` reads the region off the live simulator now, so there is nothing
    for the `State` to smuggle and the bin is an ordinary movable object like any other."""
    assert live_env.bin.type is Tossing3DEnvironment.movable_type
    for corner in ("x_min", "y_min", "z_min", "x_max", "y_max", "z_max"):
        assert corner not in live_env.bin.type.feature_names


@requires_kinder
def test_the_scene_object_carries_the_seed_set_state_would_rebuild_from(
    *, live_env: Tossing3DEnvironment
) -> None:
    """`scene` is this domain's own object, not KINDER's: it holds the two facts a flat
    `State` cannot otherwise carry, and the seed is the only thing a rewind has to go on."""
    state = live_env.get_current_state()
    assert state.get(obj=live_env.scene, feature_name="seed") == pytest.approx(CANONICAL_SEED)
    assert state.get(obj=live_env.scene, feature_name="steps_taken") == pytest.approx(0)


@requires_kinder
def test_set_state_refuses_a_mid_episode_state(*, live_env: Tossing3DEnvironment) -> None:
    """The load-bearing honesty of this domain: MuJoCo's qpos/qvel are not in a flat
    `core.State`, so there is no faithful mid-episode rewind. Quietly restoring the
    episode's *initial* state instead would make an evaluation look like it rewound."""
    mid_episode = live_env.build_state(
        kinder_state=live_env.backend().kinder_state(), seed=CANONICAL_SEED, steps_taken=2
    )
    with pytest.raises(ValueError, match="episode-initial"):
        live_env.set_state(state=mid_episode)


@requires_kinder
def test_set_state_accepts_an_episode_initial_state_and_rebuilds_from_its_seed(
    *, live_env: Tossing3DEnvironment
) -> None:
    """The other half of the same contract, which is what makes the refusal above a
    distinction rather than a blanket refusal."""
    initial = live_env.build_state(
        kinder_state=live_env.backend().kinder_state(), seed=CANONICAL_SEED, steps_taken=0
    )
    live_env.set_state(state=initial)
    assert live_env.get_current_state().get(
        obj=live_env.scene, feature_name="seed"
    ) == pytest.approx(CANONICAL_SEED)


@requires_kinder
def test_an_unknown_skill_id_is_recorded_as_a_no_op_rather_than_raising(
    *, live_env: Tossing3DEnvironment
) -> None:
    assert live_env._execute(action=np.array([7.0, 0.0, 0.0, 0.0, 0.0])) == []  # noqa: SLF001
    assert "unknown skill id: 7" in str(live_env.last_skill_error())


@requires_kinder
def test_the_noop_action_runs_no_controller_at_all(*, live_env: Tossing3DEnvironment) -> None:
    """The defect this domain surfaced: `pick_cube_id == 0`, so the `np.zeros(5)` that
    `EesMethod` used to emit when it could not plan was a real `pick_cube` at distance
    0.0 -- a whole arm trajectory. `_execute` returning no runs is exactly "no controller
    ran", which is the assertion that would have caught it."""
    assert live_env._execute(action=live_env.noop_action()) == []  # noqa: SLF001
    assert "unknown skill id" in str(live_env.last_skill_error())


@requires_kinder
def test_pick_cube_is_handed_exactly_the_first_two_parameter_slots(
    *, live_env: Tossing3DEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_execute` is where an action vector becomes controller arguments, so it is where a
    slot could be read from the wrong index -- silently, since every slot is a plausible
    float. Asserted at distinct values so a hardcoded pass-through fails here too."""
    seen: list[np.ndarray] = []
    monkeypatch.setattr(KinderBackend, "run_skill", _spy(seen=seen))

    live_env._execute(  # noqa: SLF001
        action=np.array([float(live_env.pick_cube_id), 0.41, 0.42, 9.9, 9.9])
    )

    assert len(seen) == 1
    assert seen[0] == pytest.approx([0.41, 0.42])


@requires_kinder
def test_the_fused_toss_skill_is_handed_all_four_parameter_slots_in_order(
    *, live_env: Tossing3DEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The slot ordering that the fusion made newly fragile.** These four used to arrive
    through two separate controllers with two separate signatures, so a mis-ordering could
    not silently swap a standoff for a release millisecond. They now travel as one vector
    into one `sample_parameters`, and every one of them is a bare float."""
    seen: list[np.ndarray] = []
    monkeypatch.setattr(KinderBackend, "run_skill", _spy(seen=seen))

    live_env._execute(  # noqa: SLF001
        action=np.array([float(live_env.move_to_toss_location_and_toss_id), 1.35, 0.0, 2.44, 0.72])
    )

    assert len(seen) == 1
    assert seen[0] == pytest.approx([1.35, 0.0, 2.44, 0.72])


def _spy(*, seen: list[np.ndarray]):
    """A `run_skill` stand-in that records the parameter vector and runs no simulator."""
    from hitl_pmp.adapters.kinder.types import ControllerRun

    # Positional `self` because it stands in for a bound method on KinderBackend.
    def run_skill(self: KinderBackend, **kwargs) -> ControllerRun:  # noqa: PLR0917, ANN003
        del self
        seen.append(np.asarray(kwargs["params"], dtype=float))
        return ControllerRun(steps=1, terminated=True)

    return run_skill
