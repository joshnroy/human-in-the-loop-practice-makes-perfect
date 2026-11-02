"""Opening is a release, not a guarantee of a recoverable landing."""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.core.control_steps import ControlStepLimitReached
from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.kinder_backend import (
    ClearReleaseController,
    OpenGripperSettlingController,
)
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.environments.tossing3d.predicates import (
    BIN_AT_SIDE,
    CLOSED_EMPTY,
    CUBE_AT_SIDE,
    GRASP_CLEAR,
    GRIPPER_CLOSED,
    HAND_EMPTY,
    HOLDING,
    IN_BIN,
    NOT_HOLDING,
    ON_GROUND,
    PICK_PLANNABLE,
)
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.belief_space.tossing3d_transition_model import apply_success_effects
from hitl_pmp.planning.grounding import SkillGrounder

from .observations import HOLDING_ATOMS, INITIAL_ATOMS, state
from .test_toss_settling import _controller, _State


def _open(*, provider):
    env = provider.env
    return GroundSkill(
        skill=next(skill for skill in provider.skills() if skill.name == "OpenGripper"),
        objects=(env.robot, env.cube, env.bin),
    )


def _abstract(*, provider, observed):
    return SkillGrounder.abstract_state(
        state=observed, objects=provider.objects(), predicates=provider.predicates()
    )


@pytest.mark.parametrize("layout", list(Tossing3DLayout))
@pytest.mark.parametrize("hand", ["open", "closed_empty", "holding"])
@pytest.mark.parametrize("upright", [False, True])
def test_open_applicability_and_forecast(*, layout, hand, upright):
    env = Tossing3DEnvironment(layout=layout)
    provider = Tossing3DSkillProvider(env=env)
    skill = _open(provider=provider)
    raw = HOLDING_ATOMS if hand == "holding" else INITIAL_ATOMS
    if hand == "closed_empty":
        raw = raw - {("HandEmpty", ("robot",))}
    if not upright:
        raw = raw - {("BinOnGround", ("bin_0",))}
    observed = state(env=env, abstract_atoms=raw)
    before = _abstract(provider=provider, observed=observed)
    assert (skill.preconditions <= before) == (hand != "open")
    assert skill.skill.param_dim == 0
    assert (
        provider.compute_action(ground_skill=skill, params=np.zeros(0), state=observed)[0]
        == env.open_gripper_id
        == 2
    )
    if hand == "open":
        return
    after = apply_success_effects(
        true_atoms=before,
        ground_skill=skill,
        effects={skill: (skill.add_effects, skill.delete_effects, skill.ignore_effects)},
    )
    assert skill.add_effects <= after
    assert not skill.delete_effects & after
    if hand == "holding":
        forgotten = skill.expected_ignore_effects(before_atoms=before)
        assert CUBE_AT_SIDE in forgotten
        assert not any(atom.predicate in forgotten for atom in after)
        assert not any(
            candidate.skill.name.startswith("Pick")
            for candidate in SkillGrounder.applicable_ground_skills(
                skills=provider.skills(), objects=provider.objects(), true_atoms=after
            )
        )
    else:
        assert before - skill.delete_effects <= after
    expected = (before - skill.delete_effects) - {
        atom
        for atom in before
        if atom.predicate in skill.expected_ignore_effects(before_atoms=before)
    }
    expected |= skill.expected_add_effects(before_atoms=before)
    assert after == expected


@pytest.mark.parametrize("layout", list(Tossing3DLayout))
def test_pick_reset_open_pick_forecast_preserves_empty_hand_geometry(*, layout):
    env = Tossing3DEnvironment(layout=layout)
    provider = Tossing3DSkillProvider(env=env)
    atoms = _abstract(provider=provider, observed=state(env=env, abstract_atoms=INITIAL_ATOMS))
    pick = next(
        skill
        for skill in SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
        )
        if skill.skill.name.startswith("Pick")
    )
    for skill in (pick, provider.human_cube_bin_reset_skill(), _open(provider=provider)):
        assert skill.preconditions <= atoms
        atoms = apply_success_effects(
            true_atoms=atoms,
            ground_skill=skill,
            effects={skill: (skill.add_effects, skill.delete_effects, skill.ignore_effects)},
        )
    assert GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube)) not in atoms
    assert GroundAtom(predicate=GRIPPER_CLOSED, objects=(env.robot,)) not in atoms
    assert GroundAtom(predicate=NOT_HOLDING, objects=(env.robot, env.cube)) in atoms
    assert GroundAtom(predicate=PICK_PLANNABLE, objects=(env.robot, env.cube)) in atoms
    assert pick.preconditions <= atoms


@pytest.mark.parametrize("layout", list(Tossing3DLayout))
@pytest.mark.parametrize("side", Tossing3DSides.objects())
def test_drop_forgets_stale_sides_and_geometry_without_changing_skill_identity(*, layout, side):
    env = Tossing3DEnvironment(layout=layout)
    provider = Tossing3DSkillProvider(env=env)
    opening = _open(provider=provider)
    stale = frozenset({
        GroundAtom(predicate=CUBE_AT_SIDE, objects=(env.cube, env.barrier, side)),
        GroundAtom(predicate=ON_GROUND, objects=(env.cube,)),
        GroundAtom(predicate=IN_BIN, objects=(env.cube, env.bin)),
        GroundAtom(predicate=GRASP_CLEAR, objects=(env.cube, env.bin)),
        GroundAtom(predicate=PICK_PLANNABLE, objects=(env.robot, env.cube)),
    })
    before = (
        opening.preconditions
        | stale
        | {GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube))}
    )
    after = apply_success_effects(
        true_atoms=before,
        ground_skill=opening,
        effects={opening: (opening.add_effects, opening.delete_effects, opening.ignore_effects)},
    )
    assert not stale & after
    assert after == opening.add_effects
    assert opening.objects == (env.robot, env.cube, env.bin)
    assert opening.skill.name == "OpenGripper"
    assert opening.skill.param_dim == 0


@pytest.mark.parametrize("name", ["MoveToTossLocationAndToss", "PourCubeOutOfBin"])
def test_toss_and_pour_keep_existing_functional_side_transitions(*, name):
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    lifted = next(skill for skill in provider.skills() if skill.name == name)
    objects = {
        "robot": env.robot,
        "cube": env.cube,
        "bin": env.bin,
        "barrier": env.barrier,
        "side": Tossing3DSides.opposite,
    }
    skill = GroundSkill(
        skill=lifted, objects=tuple(objects[variable.name] for variable in lifted.parameters)
    )
    previous_side = GroundAtom(
        predicate=CUBE_AT_SIDE, objects=(env.cube, env.barrier, Tossing3DSides.robot)
    )
    destination_side = GroundAtom(
        predicate=CUBE_AT_SIDE, objects=(env.cube, env.barrier, Tossing3DSides.opposite)
    )
    grasp_clear = GroundAtom(predicate=GRASP_CLEAR, objects=(env.cube, env.bin))
    pick_plannable = GroundAtom(predicate=PICK_PLANNABLE, objects=(env.robot, env.cube))
    bin_side = GroundAtom(
        predicate=BIN_AT_SIDE, objects=(env.bin, env.barrier, Tossing3DSides.opposite)
    )
    before = skill.preconditions | {previous_side, bin_side, grasp_clear, pick_plannable}
    after = apply_success_effects(
        true_atoms=before,
        ground_skill=skill,
        effects={skill: (skill.add_effects, skill.delete_effects, skill.ignore_effects)},
    )
    assert previous_side not in after
    assert {destination_side, grasp_clear, pick_plannable} <= after
    assert skill.expected_add_effects(before_atoms=before) <= after


@pytest.mark.parametrize("delta", [False, True])
@pytest.mark.parametrize("dimensions", [11, 18])
def test_open_waits_for_physics_and_holds_pose(*, delta, dimensions):
    original = _controller(delta=delta, dimensions=dimensions)
    controller = OpenGripperSettlingController(
        backend=original.backend, controller=original.controller
    )
    controller.reset(_State(), None)
    for tick in range(6):
        action = controller.step()
        assert action.shape == (dimensions,)
        np.testing.assert_allclose(action[:10], 0.0 if delta else 0.2)
        assert action[10] == 0.0
        controller.observe(_State())
        assert controller.terminated() == (tick == 5)
    assert controller.failure is None


def test_open_settle_timeout_is_not_reported_as_success():
    original = _controller()
    controller = OpenGripperSettlingController(
        backend=original.backend, controller=original.controller
    )
    controller.reset(_State(), None)
    for _ in range(21):
        controller.step()
        controller.observe(_State(speed=1.0))
    assert controller.terminated()
    assert controller.failure == "open_gripper_settle_timeout"


@pytest.mark.parametrize("angular", [False, True])
def test_open_contact_jitter_times_out_even_with_a_quiet_final_sample(*, angular):
    original = _controller()
    controller = OpenGripperSettlingController(
        backend=original.backend, controller=original.controller
    )
    controller.reset(_State(), None)
    controller.step()
    controller.observe(_State())
    for tick in range(20):
        moving = tick % 3 == 0
        controller.step()
        controller.observe(
            _State(
                speed=0.02 if moving and not angular else 0.0,
                angular_speed=0.2 if moving and angular else 0.0,
            )
        )
        assert controller.stable_ticks == tick % 3
        assert controller.terminated() == (tick == 19)
    assert controller.state.speed == controller.state.angular_speed == 0.0
    assert controller.failure == "open_gripper_settle_timeout"


@pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KINDER")
@pytest.mark.parametrize("reset_before_open", [False, True])
@pytest.mark.parametrize("layout", list(Tossing3DLayout))
@pytest.mark.parametrize("seed", [1, 125])
def test_live_pick_release_or_reset_open(*, layout, reset_before_open, seed):
    """Both the former base landing and an initially clearer grasp must settle."""
    env = Tossing3DEnvironment(layout=layout)
    provider = Tossing3DSkillProvider(env=env)
    try:
        initial = env.reset_to_seed(seed=seed)
        held = env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0]))
        assert env.last_skill_error() is None
        assert HOLDING.holds(held, (env.robot, env.cube))
        if reset_before_open:
            assert env.reset_movables(destination=Tossing3DSides.robot.name)
            held = env.get_current_state()
            assert CLOSED_EMPTY.holds(held, (env.robot, env.cube))
        before = _abstract(provider=provider, observed=held)
        opening = _open(provider=provider)
        assert opening.preconditions <= before
        after = env.take_action(
            action=provider.compute_action(ground_skill=opening, params=np.zeros(0), state=held)
        )
        assert env.last_skill_error() is None
        assert sum(env.last_controller_steps()) > 1
        observed = _abstract(provider=provider, observed=after)
        assert opening.add_effects <= observed
        assert not opening.delete_effects & observed
        assert CUBE_AT_SIDE.holds(after, (env.cube, env.barrier, Tossing3DSides.robot))
        snapshot = env.backend().live_state()
        cube = snapshot.get_object_from_name(env.backend().cube_name)
        assert np.linalg.norm([snapshot.get(cube, f"v{axis}") for axis in "xyz"]) <= 0.01
        assert np.linalg.norm([snapshot.get(cube, f"w{axis}") for axis in "xyz"]) <= 0.1
        if reset_before_open:
            assert ON_GROUND.holds(after, (env.cube,))
            assert PICK_PLANNABLE.holds(after, (env.robot, env.cube))
            picked = env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0]))
            assert HOLDING.holds(picked, (env.robot, env.cube))
        else:
            assert after.get(obj=env.cube, feature_name="z") < held.get(
                obj=env.cube, feature_name="z"
            )
        assert HAND_EMPTY.holds(initial, (env.robot,))
    finally:
        env.close()


@pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KINDER")
def test_live_release_clears_base_before_opening(*, monkeypatch):
    """The seed-125 regression must move closed, then release away from the chassis."""
    env = Tossing3DEnvironment(layout=Tossing3DLayout.SAME_SIDE)
    samples = []
    before_steps = []
    contact_bodies = set()
    try:
        env.reset_to_seed(seed=125)
        held = env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0]))
        assert env.last_skill_error() is None
        assert HOLDING.holds(held, (env.robot, env.cube))
        backend = env.kinder_backend()
        robot = backend.live_state().get_object_from_name(backend.robot_name)
        sim = backend._object_centric()._robot_env.sim
        model, data = sim.model.mj_model, sim.data.mj_data
        base_body = model.body("robot_base_link").id
        origin = data.xpos[base_body].copy()
        rotation = data.xmat[base_body].reshape(3, 3).copy()
        pinch_site = model.site("robot_pinch_site").id

        def before_step():
            before_steps.append((data.site_xpos[pinch_site].copy() - origin) @ rotation)

        def capture():
            snapshot = backend.live_state()
            cube = snapshot.get_object_from_name(backend.cube_name)
            samples.append((
                np.linalg.norm([snapshot.get(cube, f"v{axis}") for axis in "xyz"]),
                np.linalg.norm([snapshot.get(cube, f"w{axis}") for axis in "xyz"]),
                snapshot.get(robot, "pos_gripper"),
            ))
            np.testing.assert_allclose(data.xpos[base_body], origin, atol=1e-3)
            np.testing.assert_allclose(data.xmat[base_body].reshape(3, 3), rotation, atol=1e-3)
            cube_body = model.body(backend.cube_name).id
            for contact in data.contact:
                bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
                if cube_body in bodies:
                    contact_bodies.update(model.body(body).name for body in bodies)

        env.set_control_step_observers(before=before_step, after=capture)
        after = env.take_action(action=np.array([env.open_gripper_id, 0, 0, 0, 0]))
        env.set_control_step_observers(before=None, after=None)
        assert env.last_skill_error() is None
        assert "robot_base_link" not in contact_bodies
        assert HAND_EMPTY.holds(after, (env.robot,))
        assert NOT_HOLDING.holds(after, (env.robot, env.cube))
        assert ON_GROUND.holds(after, (env.cube,))
        assert after.get(obj=env.cube, feature_name="z") < held.get(obj=env.cube, feature_name="z")
        assert len(samples) == sum(env.last_controller_steps())
        opening_tick = next(index for index, sample in enumerate(samples) if sample[2] == 0)
        assert opening_tick > 0
        assert all(sample[2] > 0.2 for sample in samples[:opening_tick])
        assert before_steps[opening_tick][0] > 0.274 + ClearReleaseController.hand_radius
        assert before_steps[opening_tick][0] - before_steps[0][0] > 0.1
        assert all(linear <= 0.01 and angular <= 0.1 for linear, angular, _ in samples[-3:])
        monkeypatch.setattr(
            ClearReleaseController,
            "_plan",
            lambda *args: pytest.fail("an already-clear hand must not plan another motion"),
        )
        env.take_action(action=np.array([env.open_gripper_id, 0, 0, 0, 0]))
        assert env.last_skill_error() is None
        assert sum(env.last_controller_steps()) == 5
    finally:
        env.close()


def _clear_release(*, delta=True, dimensions=18):
    original = _controller(delta=delta, dimensions=dimensions)
    inner = OpenGripperSettlingController(backend=original.backend, controller=original.controller)
    controller = ClearReleaseController(backend=original.backend, controller=inner)
    controller.state = _State()
    controller.pose_features = tuple(f"pose_{index}" for index in range(10))
    controller.initial_pose = np.full(10, 0.2)
    controller.closed_command = 0.4
    controller.robot_env = SimpleNamespace(act_delta=delta)
    controller._opening = False
    controller._failure = None
    controller.carrying = False
    controller.motion_ticks = controller.motion_stable_ticks = controller.waypoint = 0
    controller.motion_limit = 3
    controller.path = [np.full(7, 0.2)]
    controller.fingers = []
    controller.wrap_difference = lambda difference: np.arctan2(
        np.sin(difference), np.cos(difference)
    )
    controller._refresh_planning = lambda: None
    controller._path_collision = lambda **kwargs: False
    controller._clear = lambda: False
    return controller


@pytest.mark.parametrize("delta", [False, True])
@pytest.mark.parametrize("dimensions", [11, 18])
def test_clearance_waits_closed_for_measured_clearance(*, delta, dimensions):
    controller = _clear_release(delta=delta, dimensions=dimensions)
    for tick in range(3):
        action = controller.step()
        assert action.shape == (dimensions,)
        assert action[10] == 0.4
        np.testing.assert_allclose(action[:3], 0.0 if delta else 0.2)
        controller.observe(_State())
        assert controller.terminated() == (tick == 2)
    assert controller.failure == "open_gripper_clearance_timeout"
    assert not controller._opening


@pytest.mark.parametrize("reason", ["unreachable", "blocked"])
def test_clearance_planning_failure_never_opens(*, reason):
    controller = _clear_release()
    controller._failure = f"open_gripper_clearance_{reason}"
    assert controller.step()[10] == 0.4
    assert controller.terminated()
    assert controller.failure == f"open_gripper_clearance_{reason}"
    assert not controller._opening


def test_clearance_rechecks_the_executed_segment():
    controller = _clear_release()
    controller.path = [np.full(7, 0.3)]
    controller._path_collision = lambda **kwargs: True
    action = controller.step()
    np.testing.assert_array_equal(action[:10], np.zeros(10))
    assert action[10] == 0.4
    assert controller.terminated()
    assert controller.failure == "open_gripper_clearance_blocked"


def test_clearance_starts_opening_only_after_two_stopped_clear_observations():
    controller = _clear_release()
    controller._clear = lambda: True
    for tick in range(2):
        assert controller.step()[10] == 0.4
        controller.observe(_State())
        assert controller._opening == (tick == 1)
    assert controller.step()[10] == 0.0
    assert not controller.terminated()


@pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KINDER")
def test_live_clearance_budget_interruption_leaves_gripper_closed():
    env = Tossing3DEnvironment(layout=Tossing3DLayout.SAME_SIDE)
    ticks = []
    try:
        env.reset_to_seed(seed=125)
        env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0]))
        assert env.last_skill_error() is None

        def before_step():
            if len(ticks) == 2:
                raise ControlStepLimitReached()

        env.set_control_step_observers(before=before_step, after=lambda: ticks.append(True))
        with pytest.raises(ControlStepLimitReached):
            env.take_action(action=np.array([env.open_gripper_id, 0, 0, 0, 0]))
        assert len(ticks) == 2
        snapshot = env.kinder_backend().live_state()
        robot = snapshot.get_object_from_name(env.kinder_backend().robot_name)
        assert snapshot.get(robot, "pos_gripper") > 0.2
    finally:
        env.close()


@pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KINDER")
@pytest.mark.parametrize("failure", ["blocked", "unreachable"])
def test_live_clearance_planning_refusal_keeps_cube_held(*, failure, monkeypatch):
    from kinder_models.dynamic3d import utils
    from pybullet_helpers import motion_planning
    from pybullet_helpers.inverse_kinematics import InverseKinematicsError

    env = Tossing3DEnvironment(layout=Tossing3DLayout.SAME_SIDE)
    try:
        env.reset_to_seed(seed=125)
        env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0]))
        assert env.last_skill_error() is None
        if failure == "blocked":
            monkeypatch.setattr(
                motion_planning, "run_motion_planning", lambda *args, **kwargs: None
            )
        else:

            def unreachable(*args, **kwargs):
                raise InverseKinematicsError("blocked test target")

            monkeypatch.setattr(utils, "inverse_kinematics_fixed_effort", unreachable)
        after = env.take_action(action=np.array([env.open_gripper_id, 0, 0, 0, 0]))
        assert env.last_skill_error() == f"open_gripper_clearance_{failure}"
        assert HOLDING.holds(after, (env.robot, env.cube))
        assert GRIPPER_CLOSED.holds(after, (env.robot,))
        assert sum(env.last_controller_steps()) == 1
    finally:
        env.close()
