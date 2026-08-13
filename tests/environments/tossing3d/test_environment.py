"""Offline tests for the KINDER -> `core.State` translation and the action encoding.

Everything here runs without MuJoCo. The simulator-backed half is
`test_kinder_fidelity.py`.
"""

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment

from .observations import GOAL_REGION_BBOX, observation, state


def test_constructing_the_environment_imports_no_simulator() -> None:
    """The whole reason `KinderBackend` is lazy. If constructing a `Tossing3DEnvironment`
    ever pulled MuJoCo in, CI -- which never installs the optional extra -- could not so
    much as import `hitl_pmp.cli`, since `tossing3d` is in its ENVIRONMENTS registry."""
    import sys

    env = Tossing3DEnvironment()
    assert env.variant == "o1"
    assert "mujoco" not in sys.modules


def test_translation_maps_every_declared_feature_from_the_kinder_observation() -> None:
    translated = state(cube_x=1.5, cube_y=0.25, cube_z=0.3, gripper=0.9, base_x=0.6, base_y=-0.1)
    env = Tossing3DEnvironment()

    assert translated.get(obj=env.cube, feature_name="x") == pytest.approx(1.5)
    assert translated.get(obj=env.cube, feature_name="y") == pytest.approx(0.25)
    assert translated.get(obj=env.cube, feature_name="z") == pytest.approx(0.3)
    assert translated.get(obj=env.robot, feature_name="pos_gripper") == pytest.approx(0.9)
    assert translated.get(obj=env.robot, feature_name="pos_base_x") == pytest.approx(0.6)
    assert translated.get(obj=env.robot, feature_name="pos_base_y") == pytest.approx(-0.1)


def test_the_goal_box_travels_in_the_state_rather_than_being_re_derived() -> None:
    """The predicate that decides success reads its box out of the `State`. If the box
    were re-derived from the task JSON instead it would be the *uninflated* range, which
    is 2/3 of the true width on x -- the axis a toss controls -- and every KINDER success
    near the edge would score here as a failure. That defect has shipped once already."""
    translated = state()
    env = Tossing3DEnvironment()
    corners = ("x_min", "y_min", "z_min", "x_max", "y_max", "z_max")
    read_back = tuple(translated.get(obj=env.goal_region, feature_name=name) for name in corners)
    assert read_back == pytest.approx(GOAL_REGION_BBOX)


def test_the_robot_is_found_by_its_feature_schema_not_by_a_hardcoded_name() -> None:
    """The robot's name comes from the robot config, not from the task JSON's `objects`
    block, so it is the one object this translation cannot look up by literal. It is
    identified by being the only thing carrying `pos_base_x`."""
    renamed = observation()
    renamed.features["tidybot_left"] = renamed.features.pop("robot")
    env = Tossing3DEnvironment()

    translated = env.build_state(observation=renamed, seed=125, steps_taken=0)

    assert translated.get(obj=env.robot, feature_name="pos_base_x") == pytest.approx(0.0)


def test_a_missing_object_raises_instead_of_translating_to_zeros() -> None:
    """Silence here would be a scene that changed shape upstream being read as a scene
    where everything sits at the origin."""
    incomplete = observation()
    del incomplete.features["cube_0"]
    with pytest.raises(KeyError, match="cube_0"):
        Tossing3DEnvironment().build_state(observation=incomplete, seed=0, steps_taken=0)


def test_a_missing_feature_raises_and_names_what_was_available() -> None:
    partial = observation()
    del partial.features["cube_0"]["bb_z"]
    with pytest.raises(KeyError, match="bb_z"):
        Tossing3DEnvironment().build_state(observation=partial, seed=0, steps_taken=0)


def test_set_state_refuses_a_mid_episode_state() -> None:
    """The load-bearing honesty of this domain: MuJoCo's qpos/qvel are not in a flat
    `core.State`, so there is no faithful mid-episode rewind. Quietly restoring the
    episode's *initial* state instead would make an evaluation look like it rewound."""
    env = Tossing3DEnvironment()
    mid_episode = state(env=env, steps_taken=2)
    with pytest.raises(ValueError, match="episode-initial"):
        env.set_state(state=mid_episode)


def test_the_scene_object_carries_the_seed_set_state_would_rebuild_from() -> None:
    env = Tossing3DEnvironment()
    translated = state(env=env, seed=4242)
    assert translated.get(obj=env.scene, feature_name="seed") == pytest.approx(4242)
    assert translated.get(obj=env.scene, feature_name="steps_taken") == pytest.approx(0)


def test_get_valid_actions_is_empty_because_the_parameters_are_continuous() -> None:
    assert Tossing3DEnvironment().get_valid_actions() == []


def test_the_action_space_has_one_id_slot_and_two_parameter_slots() -> None:
    assert Tossing3DEnvironment.action_space.shape == (3,)


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


def test_a_non_finite_action_is_recorded_as_a_no_op_rather_than_raising() -> None:
    """`take_action` must be total over its Box action space, and `round(inf)` raises."""
    env = Tossing3DEnvironment()
    assert env._execute(action=np.array([np.inf, 0.0, 0.0])) == []
    assert env.last_skill_error() is not None
    assert "non-finite" in env.last_skill_error()


def test_an_unknown_skill_id_is_recorded_as_a_no_op_rather_than_raising() -> None:
    env = Tossing3DEnvironment()
    assert env._execute(action=np.array([7.0, 0.0, 0.0])) == []
    assert "unknown skill id: 7" in str(env.last_skill_error())


def test_the_noop_action_runs_no_controller_at_all() -> None:
    """The defect this domain surfaced: `pick_id == 0`, so the `np.zeros(3)` that
    `EesMethod` used to emit when it could not plan was a real `pick_shelf` at
    distance 0.0. `_execute` returning no runs is exactly "no controller ran"."""
    env = Tossing3DEnvironment()
    assert env._execute(action=env.noop_action()) == []
    assert "unknown skill id" in str(env.last_skill_error())


def test_the_noop_id_is_not_a_real_skill_id() -> None:
    env = Tossing3DEnvironment()
    assert env.noop_id not in {env.pick_id, env.move_to_throw_pose_id, env.toss_id}
