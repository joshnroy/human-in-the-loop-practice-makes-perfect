"""The real backend: the sim's predicates and skills, from the robot contract."""

import importlib.util
import math

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.parameter_feasibility import TossParameterFeasibility
from hitl_pmp.environments.tossing3d.predicates import (
    BIN_AT_SIDE,
    BIN_ON_GROUND,
    CLOSED_EMPTY,
    CUBE_AT_SIDE,
    HAND_EMPTY,
    HOLDING,
    IN_BIN,
    ON_GROUND,
    PICK_PLANNABLE,
    ROBOT_AT_SIDE,
)
from hitl_pmp.environments.tossing3d.real.backend import (
    GRIPPER_OPEN_COMMAND_TOLERANCE,
    RealTossingBackend,
    RealWorld,
)
from hitl_pmp.environments.tossing3d.real.environment import RealTossing3DEnvironment
from hitl_pmp.environments.tossing3d.real.fake_robot import (
    FakeRobotPlacer,
    FakeTossingRobot,
    FakeTossOutcome,
)
from hitl_pmp.environments.tossing3d.real.geometry import RealTossGeometry
from hitl_pmp.environments.tossing3d.real.human import TerminalScenePlacer
from hitl_pmp.environments.tossing3d.real.types import (
    BarrierDisabledError,
    BarrierRect,
    RealSceneLayout,
    RealSceneSnapshot,
    RealSkillStepCosts,
    RobotObservation,
    RobotPose,
    TossingRobot,
)
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.environments.tossing3d.toss_direction import TossStandoffBand
from hitl_pmp.environments.tossing3d.types import Tossing3DBackend
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import REAL_TOSS_PARAMETERS


def _world(**robot_fields) -> tuple[FakeTossingRobot, FakeRobotPlacer, RealWorld]:
    robot = FakeTossingRobot(**robot_fields)
    placer = FakeRobotPlacer(robot=robot)
    return robot, placer, RealWorld(robot=robot, placer=placer)


def _env(*, world: RealWorld, owner: str = "practice") -> RealTossing3DEnvironment:
    env = RealTossing3DEnvironment(world=world, owner=owner)
    env.reset_to_seed(seed=3)
    return env


def _holds(*, env, predicate, objects) -> bool:
    return predicate.holds(env.get_current_state(), objects)


def _toss_skill(*, env) -> GroundSkill:
    return GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier, Tossing3DSides.opposite),
    )


def _pick(*, env) -> None:
    env.take_action(action=np.array([float(env.pick_cube_id), 0, 0, 0, 0]))


# --- the contract --------------------------------------------------------------------


def test_the_fake_and_the_backend_satisfy_their_protocols() -> None:
    robot, _, world = _world()
    assert isinstance(robot, TossingRobot)
    backend = RealTossingBackend(
        world=world,
        layout=RealTossing3DEnvironment(world=world).scene_layout,
        parameters=REAL_TOSS_PARAMETERS,
    )
    assert isinstance(backend, Tossing3DBackend)


@pytest.mark.skipif(importlib.util.find_spec("kinder_models") is None, reason="needs kinder_models")
def test_the_open_command_tolerance_is_upstreams() -> None:
    from kinder_models.dynamic3d.utils import GRIPPER_OPEN_COMMAND_TOLERANCE as upstream

    assert upstream == GRIPPER_OPEN_COMMAND_TOLERANCE


def test_a_reply_missing_a_contract_field_fails_at_the_boundary() -> None:
    robot, _, _ = _world()
    reply = robot.observe()
    del reply["cube_upright"]
    with pytest.raises(KeyError, match="cube_upright"):
        RobotObservation.parse(reply=reply)


def test_barrier_x_must_agree_with_the_barrier_rectangle() -> None:
    robot, _, _ = _world()
    reply = robot.observe()
    reply["barrier_x"] = 0.5
    with pytest.raises(ValueError, match="barrier_x"):
        RobotObservation.parse(reply=reply)


# --- the predicate mapping -----------------------------------------------------------


def test_the_initial_scene_abstracts_as_the_sims_initial_scene() -> None:
    _, placer, world = _world()
    env = _env(world=world)
    assert placer.requests()[0].kind == "scene"
    assert _holds(env=env, predicate=HAND_EMPTY, objects=(env.robot,))
    assert _holds(env=env, predicate=ON_GROUND, objects=(env.cube,))
    assert _holds(env=env, predicate=BIN_ON_GROUND, objects=(env.bin,))
    assert _holds(
        env=env,
        predicate=PICK_PLANNABLE,
        objects=(
            env.robot,
            env.cube,
        ),
    )
    assert not _holds(
        env=env,
        predicate=HOLDING,
        objects=(
            env.robot,
            env.cube,
        ),
    )
    assert not _holds(
        env=env,
        predicate=IN_BIN,
        objects=(
            env.cube,
            env.bin,
        ),
    )
    # Robot-relative sides, computed by the sim's own predicates from real positions.
    assert _holds(
        env=env,
        predicate=ROBOT_AT_SIDE,
        objects=(
            env.robot,
            env.barrier,
            Tossing3DSides.robot,
        ),
    )
    assert not _holds(
        env=env,
        predicate=ROBOT_AT_SIDE,
        objects=(
            env.robot,
            env.barrier,
            Tossing3DSides.opposite,
        ),
    )
    assert _holds(
        env=env,
        predicate=CUBE_AT_SIDE,
        objects=(
            env.cube,
            env.barrier,
            Tossing3DSides.robot,
        ),
    )
    assert _holds(
        env=env,
        predicate=BIN_AT_SIDE,
        objects=(
            env.bin,
            env.barrier,
            Tossing3DSides.opposite,
        ),
    )
    # Upstream's literal `cube.x < barrier.x`: the robot side here is +x.
    atoms = env.get_current_state().abstract_atoms
    assert ("MovableIsDownX", ("cube_0", "cuboid_barrier")) not in atoms


def test_a_cube_across_the_barrier_is_unpickable_without_asking_the_robot() -> None:
    robot, _, world = _world()
    backend = RealTossingBackend(
        world=world,
        layout=RealTossing3DEnvironment(world=world).scene_layout,
        parameters=REAL_TOSS_PARAMETERS,
    )
    robot.place(cube=RobotPose(x=-0.4, y=0.0, yaw=0.0), bin_=robot.bin_start)
    observation = backend.observe()
    assert "can_pick" not in robot.calls()
    env = RealTossing3DEnvironment(world=world)
    state = env.build_state(
        observation=observation,
        seed=3,
        steps_taken=0,
        object_centric=backend.snapshot(),
        abstract_atoms=backend.abstract_atoms(),
    )
    assert not PICK_PLANNABLE.holds(state, (env.robot, env.cube))
    assert CUBE_AT_SIDE.holds(state, (env.cube, env.barrier, Tossing3DSides.opposite))
    assert ("MovableIsDownX", ("cube_0", "cuboid_barrier")) in state.abstract_atoms


def test_after_a_pick_the_robot_is_holding_and_the_cube_is_off_the_ground() -> None:
    _, _, world = _world()
    env = _env(world=world)
    _pick(env=env)
    assert _holds(
        env=env,
        predicate=HOLDING,
        objects=(
            env.robot,
            env.cube,
        ),
    )
    assert not _holds(env=env, predicate=HAND_EMPTY, objects=(env.robot,))
    assert not _holds(env=env, predicate=ON_GROUND, objects=(env.cube,))
    assert _holds(
        env=env,
        predicate=PICK_PLANNABLE,
        objects=(
            env.robot,
            env.cube,
        ),
    )


def test_a_closed_gripper_on_nothing_is_closed_empty() -> None:
    _, _, world = _world(pick_outcome=lambda robot: False)
    env = _env(world=world)
    _pick(env=env)
    assert not _holds(env=env, predicate=HAND_EMPTY, objects=(env.robot,))
    assert not _holds(
        env=env,
        predicate=HOLDING,
        objects=(
            env.robot,
            env.cube,
        ),
    )
    assert _holds(
        env=env,
        predicate=CLOSED_EMPTY,
        objects=(
            env.robot,
            env.cube,
        ),
    )
    env.take_action(action=np.array([float(env.open_gripper_id), 0, 0, 0, 0]))
    assert _holds(env=env, predicate=HAND_EMPTY, objects=(env.robot,))


def test_an_unseen_bin_is_not_on_the_ground() -> None:
    _, _, world = _world(bin_start=None)
    env = _env(world=world)
    assert not _holds(env=env, predicate=BIN_ON_GROUND, objects=(env.bin,))


def test_the_barrier_feature_is_its_x_extent() -> None:
    _, _, world = _world(barrier=BarrierRect(yaw=math.pi / 2))
    env = _env(world=world)
    bb_x = env.get_current_state().get(obj=env.barrier, feature_name="bb_x")
    assert bb_x == pytest.approx(1.8)


# --- the toss ------------------------------------------------------------------------


def test_the_toss_draw_band_and_stand_follow_the_real_room() -> None:
    robot, _, world = _world()
    env = _env(world=world)
    _pick(env=env)
    provider = Tossing3DSkillProvider(env=env)
    state = env.get_current_state()
    skill = _toss_skill(env=env)
    rng = np.random.default_rng(0)
    accepted = 0
    for _ in range(200):
        params = provider.sample_params_at_state(ground_skill=skill, rng=rng, state=state)
        assert REAL_TOSS_PARAMETERS.violation(params=tuple(params)) is None
        if (
            provider.parameter_rejection_reason(ground_skill=skill, params=params, state=state)
            is not None
        ):
            continue
        accepted += 1
        action = provider.compute_action(ground_skill=skill, params=params, state=state)
        bin_x = state.get(obj=env.bin, feature_name="x")
        stand_x = bin_x - params[0] * math.cos(action[2])
        # The stand is on the robot's side of the barrier, footprint and all.
        assert stand_x - 0.548 / 2 > 0.025
        assert stand_x + 0.548 / 2 < 1.7
    assert accepted > 100
    del robot


def test_a_toss_sends_the_selected_stand_direction_and_retracts() -> None:
    robot, _, world = _world()
    env = _env(world=world)
    _pick(env=env)
    state = env.get_current_state()
    provider = Tossing3DSkillProvider(env=env)
    params = np.array([1.2, 120.0, 675.0])
    action = provider.compute_action(ground_skill=_toss_skill(env=env), params=params, state=state)
    env.take_action(action=action)
    toss_call = next(call for call in robot.calls() if call.startswith("toss:"))
    _, standoff, speed, release, direction = toss_call.split(":")
    assert float(standoff) == pytest.approx(1.2)
    assert float(speed) == pytest.approx(120.0)
    expected = RealTossGeometry.direction_azimuth(bin_yaw=0.0, rotation=float(action[2]))
    assert float(direction) == pytest.approx(expected, abs=1e-4)
    # The stand is east of the bin: towards the robot's (+x) side.
    assert math.cos(float(direction)) > 0
    assert robot.calls()[robot.calls().index(toss_call) + 1] == "retract"
    # The default fake scores the middle third: this toss scores.
    assert _holds(
        env=env,
        predicate=IN_BIN,
        objects=(
            env.cube,
            env.bin,
        ),
    )
    assert env.backend().check_goals()


def test_a_toss_outside_the_real_set_moves_nothing_and_charges_nothing() -> None:
    robot, _, world = _world()
    env = _env(world=world)
    _pick(env=env)
    steps: list[str] = []
    env.set_control_step_observers(before=lambda: steps.append("b"), after=None)
    env.take_action(
        action=np.array([float(env.move_to_toss_location_and_toss_id), 2.0, 0.0, 120, 600])
    )
    assert "standoff" in str(env.last_skill_error())
    assert not any(call.startswith("toss:") for call in robot.calls())
    assert steps == []
    assert _holds(
        env=env,
        predicate=HOLDING,
        objects=(
            env.robot,
            env.cube,
        ),
    )


def test_the_default_skill_step_costs_are_the_measured_sim_means() -> None:
    costs = RealSkillStepCosts()
    assert costs.pick_cube == 119
    assert costs.move_to_toss_location_and_toss == 90
    assert costs.open_gripper == 1


def test_each_skill_is_charged_its_average_sim_tick_count() -> None:
    """One `before` -- an exhausted budget still stops a skill from starting -- then one
    `after` per sim tick the skill takes on average, as the sim's step protocol counts."""
    _, _, world = _world()
    env = _env(world=world)
    costs = RealSkillStepCosts()
    steps: list[str] = []
    env.set_control_step_observers(
        before=lambda: steps.append("before"), after=lambda: steps.append("after")
    )
    _pick(env=env)
    assert steps == ["before"] + ["after"] * costs.pick_cube
    steps.clear()
    provider = Tossing3DSkillProvider(env=env)
    action = provider.compute_action(
        ground_skill=_toss_skill(env=env),
        params=np.array([1.2, 120.0, 675.0]),
        state=env.get_current_state(),
    )
    env.take_action(action=action)
    assert steps == ["before"] + ["after"] * costs.move_to_toss_location_and_toss
    steps.clear()
    env.take_action(action=np.array([float(env.open_gripper_id), 0, 0, 0, 0]))
    assert steps == ["before"] + ["after"] * costs.open_gripper


def test_the_skill_step_costs_are_configurable() -> None:
    _, _, world = _world()
    env = RealTossing3DEnvironment(
        world=world,
        step_costs=RealSkillStepCosts(
            pick_cube=7, move_to_toss_location_and_toss=5, open_gripper=2
        ),
    )
    env.reset_to_seed(seed=3)
    steps: list[str] = []
    env.set_control_step_observers(before=None, after=lambda: steps.append("after"))
    _pick(env=env)
    assert len(steps) == 7


def test_a_missed_toss_leaves_the_cube_where_it_landed() -> None:
    _, _, world = _world(
        toss_outcome=lambda standoff, speed, release, direction, robot: FakeTossOutcome(
            landed=(-0.2, 0.9), in_bin=False
        )
    )
    env = _env(world=world)
    _pick(env=env)
    provider = Tossing3DSkillProvider(env=env)
    action = provider.compute_action(
        ground_skill=_toss_skill(env=env),
        params=np.array([1.2, 120.0, 675.0]),
        state=env.get_current_state(),
    )
    env.take_action(action=action)
    assert not _holds(
        env=env,
        predicate=IN_BIN,
        objects=(
            env.cube,
            env.bin,
        ),
    )
    assert _holds(
        env=env,
        predicate=CUBE_AT_SIDE,
        objects=(
            env.cube,
            env.barrier,
            Tossing3DSides.opposite,
        ),
    )
    assert not _holds(
        env=env,
        predicate=PICK_PLANNABLE,
        objects=(
            env.robot,
            env.cube,
        ),
    )


def test_the_barrier_is_uncrossable_for_a_stand_round_its_end() -> None:
    robot, _, world = _world()
    env = _env(world=world)
    snapshot = env.backend().snapshot()
    assert isinstance(snapshot, RealSceneSnapshot)
    geometry = RealTossGeometry.of(snapshot=snapshot)
    assert geometry is not None
    # Bin at (-0.8, 0); a 1.2 m stand south of it, at (-0.8, -1.2), is in the room and
    # clear of the barrier's rectangle (which ends at y = -0.9), but across its line.
    reason = TossParameterFeasibility.rejection_reason_from_geometry(
        geometry=geometry, params=np.array([1.2, math.pi / 2])
    )
    assert reason is not None and "uncrossable" in reason
    del robot


def test_the_real_band_is_the_four_direction_scan_inside_the_real_set() -> None:
    _, _, world = _world()
    env = _env(world=world)
    _pick(env=env)
    snapshot = env.backend().snapshot()
    geometry = RealTossGeometry.of(snapshot=snapshot)
    low, high = TossStandoffBand.robot_side_bounds(
        geometry=geometry, parameters=REAL_TOSS_PARAMETERS
    )
    assert REAL_TOSS_PARAMETERS.standoff_bounds[0] <= low < high
    assert high <= REAL_TOSS_PARAMETERS.standoff_bounds[1]


# --- the human reset and the shared world -------------------------------------------


def test_a_human_reset_goes_home_and_requests_the_sides_region() -> None:
    robot, placer, world = _world()
    env = _env(world=world)
    env.reset_movables(destination="robot_side")
    request = placer.requests()[-1]
    assert request.kind == "reset"
    assert request.destination == "robot_side"
    assert request.bin_region == env.scene_layout.robot_side_bin_region
    assert "go_home" in robot.calls()
    assert _holds(
        env=env,
        predicate=BIN_AT_SIDE,
        objects=(
            env.bin,
            env.barrier,
            Tossing3DSides.robot,
        ),
    )


def test_the_ceiling_camera_overlap_strip_is_a_named_layout_value() -> None:
    assert RealSceneLayout().ceiling_camera_overlap_y == (-0.15, 0.18)


def _in_strip(*, y: float, layout: RealSceneLayout) -> bool:
    assert layout.ceiling_camera_overlap_y is not None
    low, high = layout.ceiling_camera_overlap_y
    return low <= y <= high


def test_scene_and_reset_cube_targets_avoid_the_overlap_strip() -> None:
    _, placer, world = _world()
    env = _env(world=world)
    layout = env.scene_layout
    region = layout.cube_reset_region
    backend = env.real_backend()
    targets = [backend.scene_request(seed=seed).cube_target for seed in range(300)]
    for _ in range(100):
        env.reset_movables(destination="robot_side")
        targets.append(placer.requests()[-1].cube_target)
    for target in targets:
        assert target is not None
        x, y = target
        assert region.x_min <= x <= region.x_max
        assert region.y_min <= y <= region.y_max
        assert not _in_strip(y=y, layout=layout)
    # Both sides of the strip are used: it is cut out, not a side chosen.
    assert any(y < -0.15 for _, y in targets) and any(y > 0.18 for _, y in targets)


def test_the_prompt_tells_the_person_to_keep_the_cube_out_of_the_strip() -> None:
    _, placer, world = _world()
    env = _env(world=world)
    env.reset_movables(destination="robot_side")
    request = placer.requests()[-1]
    assert request.cube_avoid_y == (-0.15, 0.18)
    text = "\n".join(TerminalScenePlacer.describe(request=request))
    assert "-0.15" in text and "0.18" in text and "overlap" in text


def test_the_strip_does_not_move_the_bin_targets() -> None:
    """The cube's draw uses as much randomness as before, so a seed's bin target is
    the one it always was."""
    open_layout = RealSceneLayout(ceiling_camera_overlap_y=None)
    _, _, world = _world()
    env = _env(world=world)
    open_env = RealTossing3DEnvironment(world=world, scene_layout=open_layout)
    for seed in range(20):
        with_strip = env.real_backend().scene_request(seed=seed)
        without = open_env.real_backend().scene_request(seed=seed)
        assert with_strip.bin_target == without.bin_target
        assert without.cube_avoid_y is None


def test_a_cube_the_robot_left_in_the_strip_is_still_picked() -> None:
    """No pick constraint: a toss or drop into the strip is picked as anywhere else, so
    a perception problem there shows up as missed picks, not as a refusal."""
    robot, _, world = _world()
    env = _env(world=world)
    robot.place(cube=RobotPose(x=0.8, y=0.0, yaw=0.0), bin_=robot._bin)  # noqa: SLF001
    world.moved()
    env.backend().observe()
    env._adopt(  # noqa: SLF001
        state=env._observed_state(seed=3, steps_taken=0)  # noqa: SLF001
    )
    assert _in_strip(y=0.0, layout=env.scene_layout)
    assert _holds(env=env, predicate=PICK_PLANNABLE, objects=(env.robot, env.cube))
    _pick(env=env)
    assert "pick" in robot.calls()
    assert _holds(env=env, predicate=HOLDING, objects=(env.robot, env.cube))


def test_evaluation_between_practice_steps_is_followed_by_a_restore() -> None:
    robot, placer, world = _world()
    practice = _env(world=world, owner="practice")
    _pick(env=practice)
    practice_bin = practice.get_current_state().get(obj=practice.bin, feature_name="x")
    evaluation = _env(world=world, owner="evaluation")
    del evaluation
    _pick(env=practice)
    restore = placer.requests()[-1]
    assert restore.kind == "restore"
    assert restore.owner == "practice"
    assert restore.was_holding
    assert restore.bin_target is not None
    assert restore.bin_target[0] == pytest.approx(practice_bin)
    assert world.events()[-1]["kind"] == "restore"
    del robot


# --- the robot side's contract additions (2026-10-15) --------------------------------


class SkillStopped(RuntimeError):
    """Stands in for the client's own: a `stop()` interrupted a motion skill."""


class SkillBusy(RuntimeError):
    """Stands in for the client's own: another motion skill was already running."""


@pytest.mark.parametrize(
    ("error", "charged"), [(SkillStopped, RealSkillStepCosts().pick_cube), (SkillBusy, 0)]
)
def test_an_interrupted_or_refused_skill_is_a_failed_action_not_a_retry(
    *, error: type[RuntimeError], charged: int
) -> None:
    def interrupt(skill: str) -> None:  # noqa: PLR0917 - the fake's positional hook
        if skill == "pick":
            raise error("interrupted")

    robot, _, world = _world(interrupt=interrupt)
    env = _env(world=world)
    steps: list[str] = []
    env.set_control_step_observers(before=None, after=lambda: steps.append("after"))
    _pick(env=env)
    assert env.last_skill_error() == f"aborted: {error.__name__}: interrupted"
    assert robot.calls().count("pick") == 0
    assert len(steps) == charged
    assert world.events()[-1]["event"] == "skill_aborted"


def test_a_disabled_barrier_is_refused_by_name() -> None:
    _, _, world = _world(barrier=None)
    with pytest.raises(BarrierDisabledError, match="Enable the barrier"):
        _env(world=world)


def test_a_toss_while_not_holding_moves_nothing_and_does_not_retract() -> None:
    robot, _, world = _world()
    env = _env(world=world)
    env.take_action(
        action=np.array([float(env.move_to_toss_location_and_toss_id), 1.2, 0.0, 120, 675])
    )
    assert "not holding" in str(env.last_skill_error())
    assert "retract" not in robot.calls()


@pytest.mark.parametrize("in_bin", [True, False])
def test_a_cube_hidden_by_the_bin_after_a_toss_is_asked_about_not_called_a_miss(
    *, in_bin: bool
) -> None:
    robot, placer, world = _world(
        toss_outcome=lambda standoff, speed, release, direction, robot: FakeTossOutcome(
            landed=(robot.bin_start.x, robot.bin_start.y), in_bin=in_bin, visible=False
        )
    )
    env = _env(world=world)
    _pick(env=env)
    action = Tossing3DSkillProvider(env=env).compute_action(
        ground_skill=_toss_skill(env=env),
        params=np.array([1.2, 120.0, 675.0]),
        state=env.get_current_state(),
    )
    env.take_action(action=action)
    assert placer.inspections() == 1
    assert _holds(env=env, predicate=IN_BIN, objects=(env.cube, env.bin)) == in_bin
    assert env.backend().check_goals() == in_bin
    # Not in the bin and unseen: no side, so only the human reset recovers it.
    assert (
        _holds(
            env=env,
            predicate=CUBE_AT_SIDE,
            objects=(env.cube, env.barrier, Tossing3DSides.opposite),
        )
        == in_bin
    )
    assert world.events()[-1]["event"] == "hidden_cube_inspected"
    del robot


def test_one_skill_costs_one_observe() -> None:
    robot, _, world = _world()
    env = _env(world=world)
    before = robot.calls().count("observe")
    _pick(env=env)
    assert robot.calls().count("observe") == before + 1
