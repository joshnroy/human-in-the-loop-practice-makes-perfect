"""The real backend under the sim's own environment, predicates and toss pipeline."""

import logging
import math
import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.predicates import (
    BIN_ON_OPPOSITE_SIDE,
    BIN_ON_ROBOT_SIDE,
    CUBE_ON_OPPOSITE_SIDE,
    CUBE_ON_ROBOT_SIDE,
)
from hitl_pmp.environments.tossing3d.real.backend import (
    PlacementRequest,
    PracticeScene,
    RealTossingBackend,
    TerminalPerson,
    WallClock,
)
from hitl_pmp.environments.tossing3d.real.environment import RealTossing3DEnvironment
from hitl_pmp.environments.tossing3d.real.fake_robot import (
    FAKE_SKILL_SECONDS,
    FakePerson,
    FakeTossingRobot,
)
from hitl_pmp.environments.tossing3d.real.frame import BIN_SPOTS, CUBE_SPOT, ROOM_FRAME
from hitl_pmp.environments.tossing3d.real.session_log import RealSessionLog
from hitl_pmp.environments.tossing3d.sides import Tossing3DSide
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.environments.tossing3d.toss import Tossing3DToss
from hitl_pmp.environments.tossing3d.types import Tossing3DObservation
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import REAL_TOSS_PARAMETERS
from hitl_pmp.humans.oracle import UnconditionalHumanOracle

MOTION = {"pick", "toss", "open_gripper", "retract", "go_home"}


def _env(*, written: list[str] | None = None) -> tuple[RealTossing3DEnvironment, FakeTossingRobot]:
    robot = FakeTossingRobot()
    lines = written if written is not None else []
    person = FakePerson(robot=robot, write=lines.append)
    env = RealTossing3DEnvironment(tossing_robot=robot, person=person, clock=robot.now)
    env.hard_reset()
    return env, robot


def _side(*, env: RealTossing3DEnvironment, obj: object) -> set[str]:
    state = env.get_current_state()
    robot, opposite = (
        (CUBE_ON_ROBOT_SIDE, CUBE_ON_OPPOSITE_SIDE)
        if obj == env.cube
        else (BIN_ON_ROBOT_SIDE, BIN_ON_OPPOSITE_SIDE)
    )
    return {
        name
        for name, predicate in (("robot_side", robot), ("opposite_side", opposite))
        if predicate.holds(state=state, objects=(obj, env.barrier))
    }


def test_the_task_scene_is_the_cube_at_its_spot_and_the_bin_across_the_tape() -> None:
    written: list[str] = []
    env, _ = _env(written=written)
    state = env.get_current_state()
    cube_xy = ROOM_FRAME.to_domain(x=CUBE_SPOT[0], y=CUBE_SPOT[1])
    assert state.get(obj=env.cube, feature_name="x") == pytest.approx(cube_xy[0])
    assert state.get(obj=env.barrier, feature_name="x") == pytest.approx(0.0, abs=1e-3)
    assert state.get(obj=env.barrier, feature_name="bb_x") == pytest.approx(0.05)
    assert _side(env=env, obj=env.cube) == {"robot_side"}
    assert _side(env=env, obj=env.bin) == {"opposite_side"}
    atoms = {name for name, _ in state.abstract_atoms or ()}
    assert {"HandEmpty", "OnGround", "MovableIsDownX", "BinOnGround", "PickPlannable"} <= atoms
    prompt = "\n".join(written)
    assert "scene: FAR" in prompt
    assert "0.24, 1.40" not in prompt  # coordinates go to the log file only


@pytest.mark.parametrize(
    ("destination", "expected_side", "spot_name"),
    [("robot_side", "robot_side", "SAME"), ("opposite_side", "opposite_side", "FAR")],
)
def test_a_human_reset_prompts_for_the_named_spot_moves_no_robot_and_reobserves(
    *, destination: str, expected_side: str, spot_name: str, caplog: pytest.LogCaptureFixture
) -> None:
    written: list[str] = []
    env, robot = _env(written=written)
    robot.cube = None  # lost: only a reset recovers it
    written.clear()
    robot.calls.clear()
    logger = RealSessionLog.LOGGER
    level = logger.level
    logger.addHandler(caplog.handler)
    logger.setLevel(logging.DEBUG)
    caplog.handler.setLevel(logging.DEBUG)
    try:
        UnconditionalHumanOracle.execute_movables_reset(env=env, destination=destination)
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(level)
    spot = BIN_SPOTS[Tossing3DSide(destination)]
    prompt = "\n".join(written)
    (request,) = [line for line in written if line.startswith("human reset:")]
    assert request.startswith(f"human reset: {spot_name} ")
    assert "\n" not in request
    assert "cube anywhere on our side" in request
    coordinates = f"{spot.center[0]:.2f}, {spot.center[1]:.2f}"
    assert coordinates not in prompt
    assert coordinates in caplog.text  # the log file keeps the coordinates
    assert "in place? [y/n]" in prompt
    assert not MOTION & {name for name, _ in robot.calls}
    assert robot.calls[0][0] == "observe"  # only after the person is done
    assert _side(env=env, obj=env.bin) == {expected_side}
    assert _side(env=env, obj=env.cube) == {"robot_side"}


def _toss_skill(*, env: RealTossing3DEnvironment) -> GroundSkill:
    return GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier),
    )


@pytest.mark.parametrize(
    ("destination", "map_direction"), [("robot_side", 0.0), ("opposite_side", -math.pi / 2)]
)
def test_the_real_toss_is_thrown_from_the_fixed_direction_of_the_bins_side(
    *, destination: str, map_direction: float
) -> None:
    env, robot = _env()
    env.reset_movables(destination=destination)
    env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0], dtype=float))
    state = env.get_current_state()
    skill = _toss_skill(env=env)
    bounds = Tossing3DToss.standoff_bounds(
        ground_skill=skill, state=state, parameters=REAL_TOSS_PARAMETERS
    )
    assert bounds == REAL_TOSS_PARAMETERS.standoff_bounds
    params = np.array([1.20, 120.0, 675.0])
    assert (
        Tossing3DToss.rejection_reason(
            ground_skill=skill, state=state, params=params, parameters=REAL_TOSS_PARAMETERS
        )
        is None
    )
    action = Tossing3DToss.compute_action(params=params, state=state)
    env.take_action(action=action)
    tosses = [args for name, args in robot.calls if name == "toss"]
    assert len(tosses) == 1
    standoff, speed, release_ms, direction = tosses[0]
    assert (standoff, speed, release_ms) == (1.20, 120.0, 675.0)
    assert math.remainder(direction - map_direction, math.tau) == pytest.approx(0.0, abs=1e-9)


def test_a_toss_outside_the_real_set_is_refused_without_calling_the_robot() -> None:
    env, robot = _env()
    env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0], dtype=float))
    robot.calls.clear()
    env.take_action(action=np.array([env.move_to_toss_location_and_toss_id, 2.4, 0, 300, 450]))
    assert "toss" not in {name for name, _ in robot.calls}
    assert env.last_controller_steps() == (0,)
    assert "outside" in (env.last_skill_error() or "")


def test_a_server_refusal_is_an_uncharged_failed_toss() -> None:
    env, robot = _env()
    env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0], dtype=float))
    # A bin placed past its spot puts the stand across the tape: the robot side refuses.
    robot.bin = (0.24, 1.60, 0.0)
    env.take_action(
        action=np.array([env.move_to_toss_location_and_toss_id, 1.2, -math.pi / 2, 120, 675])
    )
    assert env.last_controller_steps() == (0,)
    assert "refused" in (env.last_skill_error() or "")
    assert robot.holding


def test_a_real_skill_is_charged_its_seconds_as_sim_ticks_after_one_before_check() -> None:
    env, _ = _env()
    counts = {"before": 0, "after": 0}

    def before() -> None:
        counts["before"] += 1

    def after() -> None:
        counts["after"] += 1

    env.set_control_step_observers(before=before, after=after)
    env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0], dtype=float))
    assert counts == {"before": 1, "after": round(FAKE_SKILL_SECONDS["pick"] / 0.1)}


def test_a_timed_real_skill_uses_up_its_measured_seconds_as_sim_ticks(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A skill that moved -- held, missed or aborted mid-motion -- is charged its own
    wall-clock seconds in sim ticks, as a sim skill is charged the ticks it ran; a
    refused one moved nothing and stays uncharged."""
    clock = iter([100.0, 136.4, 200.0, 207.25, 300.0, 300.5])
    monkeypatch.setattr(WallClock, "now", staticmethod(lambda: next(clock)))
    robot = FakeTossingRobot()
    env = RealTossing3DEnvironment(tossing_robot=robot, person=FakePerson(robot=robot))
    env.hard_reset()
    env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0], dtype=float))
    assert env.last_controller_steps() == (364,)

    def lost(self: FakeTossingRobot) -> dict[str, object]:  # noqa: PLR0917
        raise ConnectionError("lost")

    with monkeypatch.context() as patch:
        patch.setattr(FakeTossingRobot, "pick", lost)
        env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0], dtype=float))
    assert env.last_controller_steps() == (72,)
    assert "aborted" in (env.last_skill_error() or "")
    robot.cube = (0.24, 1.40, 0.0)  # across the tape: the robot side refuses
    env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0], dtype=float))
    assert env.last_controller_steps() == (0,)


def test_a_reply_without_a_barrier_is_refused() -> None:
    env, robot = _env()
    robot.barrier_enabled = False
    with pytest.raises(ValueError, match="barrier"):
        env.reset_movables(destination="robot_side")


def test_a_refused_pick_moved_nothing_and_is_not_charged() -> None:
    env, robot = _env()
    robot.cube = (0.24, 1.40, 0.0)  # across the tape: the robot side refuses
    env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0], dtype=float))
    assert env.last_controller_steps() == (0,)
    assert "refused" in (env.last_skill_error() or "")


def test_the_fixed_direction_is_the_one_the_robot_reports_for_the_bins_spot() -> None:
    env, robot = _env()
    snapshot = env.get_current_state().object_centric
    reported = robot.observe()["toss_direction"]
    assert snapshot.toss_azimuth() == pytest.approx(ROOM_FRAME.yaw_to_domain(yaw=reported))


class _Asked(FakePerson):
    """A FakePerson who records each confirmation and answers from a script."""

    answers: list[str] = []
    asked: list[str] = []

    def model_post_init(self, __context: object) -> None:
        super().model_post_init(__context)
        self.read = lambda prompt: self._answer(prompt=prompt)

    def _answer(self, *, prompt: str) -> str:
        if "run it?" not in prompt:
            return "y"
        self.asked.append(prompt)
        return self.answers.pop(0)


def test_every_skill_waits_for_the_persons_y_before_the_robot_moves() -> None:
    robot = FakeTossingRobot()
    person = _Asked(robot=robot, write=lambda _line: None, answers=["y", "Y"], asked=[])
    env = RealTossing3DEnvironment(tossing_robot=robot, person=person, confirm_each_skill=True)
    env.hard_reset()
    env.backend().run_pick_cube()
    env.backend().run_open_gripper()
    assert len(person.asked) == 2
    assert "PickCube" in person.asked[0] and "OpenGripper" in person.asked[1]


def test_anything_but_y_stops_the_run_before_the_robot_moves() -> None:
    robot = FakeTossingRobot()
    person = _Asked(robot=robot, write=lambda _line: None, answers=["n"], asked=[])
    env = RealTossing3DEnvironment(tossing_robot=robot, person=person, confirm_each_skill=True)
    env.hard_reset()
    before = robot.observe()["robot"]
    with pytest.raises(KeyboardInterrupt):
        env.backend().run_pick_cube()
    assert robot.observe()["robot"] == before and not robot.observe()["holding"]


def test_without_confirmation_no_skill_asks() -> None:
    robot = FakeTossingRobot()
    person = _Asked(robot=robot, write=lambda _line: None, answers=[], asked=[])
    env = RealTossing3DEnvironment(tossing_robot=robot, person=person)
    env.hard_reset()
    env.backend().run_pick_cube()
    assert person.asked == []


def test_a_task_scene_drives_home_first_and_a_human_reset_does_not() -> None:
    robot = FakeTossingRobot()
    person = _Asked(robot=robot, write=lambda _line: None, answers=["y"], asked=[])
    env = RealTossing3DEnvironment(tossing_robot=robot, person=person, confirm_start=True)
    env.hard_reset()
    names = [name for name, _ in robot.calls]
    assert "go_home" in names
    assert names.index("go_home") < names.index("observe")
    assert person.asked and "GoHome" in person.asked[0]
    robot.calls.clear()
    UnconditionalHumanOracle.execute_movables_reset(env=env, destination="robot_side")
    assert "go_home" not in [name for name, _ in robot.calls]


def test_the_person_is_told_the_robot_is_busy_after_each_answer() -> None:
    lines: list[str] = []
    robot = FakeTossingRobot()
    person = _Asked(robot=robot, write=lines.append, answers=[], asked=[])
    env = RealTossing3DEnvironment(tossing_robot=robot, person=person)
    env.hard_reset()
    env.backend().run_pick_cube()
    said = "\n".join(lines)
    for line in (
        "driving home",
        "observing the scene",
        "planning the next skill",
        "running PickCube",
        "PickCube: ok",
    ):
        assert line in said


def test_by_default_a_real_run_asks_only_at_the_start_and_at_placements() -> None:
    robot = FakeTossingRobot()
    person = _Asked(robot=robot, write=lambda _line: None, answers=["y"], asked=[])
    env = RealTossing3DEnvironment(tossing_robot=robot, person=person, confirm_start=True)
    env.hard_reset()
    env.backend().run_pick_cube()
    env.backend().run_open_gripper()
    assert len(person.asked) == 1 and "GoHome" in person.asked[0]


def test_a_scene_is_looked_at_once_until_something_happens() -> None:
    env, robot = _env()
    robot.calls.clear()
    env.backend().observe()
    env.backend().observe()
    env.get_current_state()
    assert [name for name, _ in robot.calls].count("observe") == 0
    env.backend().run_open_gripper()  # a held pick needs no look; any other skill looks
    names = [name for name, _ in robot.calls]
    assert names.count("observe") == 1
    robot.calls.clear()
    env.backend().observe()
    assert robot.calls == []


class _OneLookRobot(FakeTossingRobot):
    """A server whose observe reply already carries its can_pick verdict (API v3)."""

    def observe(self) -> dict[str, object]:
        reply = super().observe()
        reply["can_pick"] = {"feasible": True, "note": None, "refused": False, "reason": None}
        return reply


def test_a_can_pick_in_the_observe_reply_saves_the_second_camera_round() -> None:
    robot = _OneLookRobot()
    person = FakePerson(robot=robot, write=lambda _line: None)
    env = RealTossing3DEnvironment(tossing_robot=robot, person=person)
    env.hard_reset()
    assert "can_pick" not in [name for name, _ in robot.calls]
    atoms = {name for name, _ in env.get_current_state().abstract_atoms or ()}
    assert "PickPlannable" in atoms


def test_only_y_or_n_answers_anything_else_asks_again() -> None:
    lines: list[str] = []
    answers = iter(["yy", "ok", "Yes", "y"])
    person = TerminalPerson(write=lines.append, read=lambda _prompt: next(answers))
    person.confirm(action="PickCube")
    assert sum("please type y or n" in line for line in lines) == 2
    # A blank answer (a stray Enter) asks again rather than ending the run.
    declines = iter(["", "  ", "No"])
    person = TerminalPerson(write=lines.append, read=lambda _prompt: next(declines))
    with pytest.raises(KeyboardInterrupt):
        person.confirm(action="PickCube")


def test_a_timed_human_reset_is_the_seconds_between_the_start_y_and_the_done_y() -> None:
    """With `time_human_resets`, a human reset asks for a y when the person starts and
    a y when it is done; the seconds between them are the reset's measured time. A
    scene or a restore is the experiment's own setup: one y, never timed."""
    prompts: list[str] = []
    ticks = iter([100.0, 140.5])

    person = TerminalPerson(
        write=lambda _line: None,
        read=lambda prompt: prompts.append(prompt) or "y",
        time_human_resets=True,
        clock=lambda: next(ticks),
    )
    request = PlacementRequest(
        kind="human reset",
        destination=Tossing3DSide.ROBOT,
        cube=CUBE_SPOT,
        bin=BIN_SPOTS[Tossing3DSide.ROBOT],
    )
    person.place(request=request)
    assert len(prompts) == 2 and "start" in prompts[0].lower()
    assert "in place" in prompts[1]
    assert person.drain_seconds() == pytest.approx(40.5)
    assert person.drain_seconds() is None
    prompts.clear()
    person.place(request=request.model_copy(update={"kind": "scene"}))
    assert len(prompts) == 1 and person.drain_seconds() is None


def test_an_evaluation_reset_after_practice_ends_holding_leaves_the_cube_pickable() -> None:
    """Run 9 (2026-10-20): practice's last action was a pick, so the robot still held
    the cube; evaluation's scene reset drove home with the gripper still commanded
    closed, `HandEmpty` never held, and the planner found no plan at step 0."""
    practice, robot = _env()
    practice.take_action(action=np.array([practice.pick_cube_id, 0, 0, 0, 0], dtype=float))
    assert robot.holding and robot.gripper_command == 1.0
    written: list[str] = []
    # Evaluation builds its own problem over the same, one robot.
    evaluation = RealTossing3DEnvironment(
        tossing_robot=robot, person=FakePerson(robot=robot, write=written.append), evaluation=True
    )
    robot.calls.clear()
    evaluation.backend().reset(seed=0)
    atoms = {name for name, _ in evaluation.backend().abstract_atoms()}
    assert {"HandEmpty", "OnGround", "PickPlannable"} <= atoms
    assert "Holding" not in atoms
    names = [name for name, _ in robot.calls]
    assert names.index("go_home") < names.index("open_gripper") < names.index("observe")
    (request,) = [line for line in written if line.startswith("scene:")]
    assert "gripper" in request  # the person is told to take a held cube out


def test_a_scene_reset_after_seeing_the_cube_held_says_to_take_it_out() -> None:
    written: list[str] = []
    env, _ = _env(written=written)
    env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0], dtype=float))
    written.clear()
    env.backend().reset(seed=0)
    (request,) = [line for line in written if line.startswith("scene:")]
    assert "was holding the cube" in request
    written.clear()
    env.backend().reset(seed=0)  # last seen with the gripper empty: nothing to take out
    (request,) = [line for line in written if line.startswith("scene:")]
    assert "gripper" not in request


def test_each_skill_is_preceded_by_a_progress_line_for_the_practice_cycle() -> None:
    written: list[str] = []
    env, _ = _env(written=written)
    written.clear()
    env.backend().practice_progress(cycle=0, action=13, max_actions=20, steps=1050, budget=1500)
    env.backend().run_pick_cube()
    progress = written.index("cycle 0 [######----] action 13/20 · steps 1050/1500")
    assert written[progress + 1].startswith("running PickCube")


def test_an_evaluation_skill_has_no_practice_progress_line() -> None:
    written: list[str] = []
    env, _ = _env(written=written)
    written.clear()
    env.backend().run_pick_cube()
    assert not any(line.startswith("cycle ") for line in written)


STAMP = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} "


def test_every_line_and_prompt_is_timestamped_on_the_terminal_and_in_the_session_log(
    *, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    RealSessionLog.configure(output_dir=tmp_path)
    try:
        prompts: list[str] = []

        def typed(prompt: str) -> str:  # noqa: PLR0917 -- `input`'s signature
            prompts.append(prompt)
            return "y"

        monkeypatch.setattr("builtins.input", typed)
        person = TerminalPerson()
        person.write("driving home ...")
        person.confirm(action="PickCube")
        shown = capsys.readouterr().out
        assert re.search(f"^{STAMP}driving home \\.\\.\\.$", shown, flags=re.MULTILINE)
        (prompt,) = prompts
        assert re.fullmatch(f"{STAMP}PickCube -- run it\\? \\[y/n\\] ", prompt)
        logged = (tmp_path / RealSessionLog.FILENAME).read_text()
        assert re.search(f"^{STAMP}driving home \\.\\.\\.$", logged, flags=re.MULTILINE)
        assert re.search(f"^{STAMP}PickCube -- run it\\? \\[y/n\\]$", logged, flags=re.MULTILINE)
        assert re.search(f"^{STAMP}answer: 'y'$", logged, flags=re.MULTILINE)
    finally:
        RealSessionLog.close()


# --- evaluation after each cycle: the practice scene is remembered and restored ---------


def test_the_practice_scene_remembers_the_cubes_map_spot_and_the_bins_side() -> None:
    env, _ = _env()
    scene = env.backend().practice_scene()
    assert scene.cube == pytest.approx(CUBE_SPOT)
    assert scene.bin_side is Tossing3DSide.OPPOSITE
    assert not scene.holding


def test_restoring_practice_drives_home_opens_the_gripper_and_asks_for_the_scene() -> None:
    written: list[str] = []
    env, robot = _env(written=written)
    scene = PracticeScene(cube=(0.66, -0.98), holding=False, bin_side=Tossing3DSide.ROBOT)
    robot.calls.clear()
    written.clear()
    env.restore_practice_scene(scene=scene)
    names = [name for name, _ in robot.calls]
    assert names.index("go_home") < names.index("open_gripper") < names.index("observe")
    (request,) = [line for line in written if line.startswith("restore practice:")]
    assert request == (
        "restore practice: SAME (bin at the close spot, our side of the tape); "
        "cube at (0.66, -0.98) on our side, on a face; do not move the robot"
    )
    # Practice carries on from the scene as placed, re-observed.
    assert robot.cube is not None and robot.cube[:2] == pytest.approx((0.66, -0.98))
    state = env.get_current_state()
    cube_x, _ = ROOM_FRAME.to_domain(x=0.66, y=-0.98)
    assert state.get(obj=env.cube, feature_name="x") == pytest.approx(cube_x)
    assert _side(env=env, obj=env.bin) == {"robot_side"}


def test_a_practice_scene_holding_the_cube_asks_for_it_on_the_floor_where_the_robot_was() -> None:
    written: list[str] = []
    env, robot = _env(written=written)
    env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0], dtype=float))
    scene = env.backend().practice_scene()
    assert scene.holding
    assert scene.cube == pytest.approx(robot.robot[:2])
    written.clear()
    env.restore_practice_scene(scene=scene)
    (request,) = [line for line in written if line.startswith("restore practice:")]
    assert "holding the cube" in request and "where the robot was" in request
    atoms = {name for name, _ in env.get_current_state().abstract_atoms or ()}
    assert {"HandEmpty", "OnGround"} <= atoms
    assert "Holding" not in atoms


def test_restoring_practice_costs_no_steps_and_keeps_the_episode_bookkeeping() -> None:
    env, _ = _env()
    env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0], dtype=float))
    before = env.get_current_state().get(obj=env.scene, feature_name="steps_taken")
    charged: list[str] = []
    env.set_control_step_observers(
        before=lambda: charged.append("before"), after=lambda: charged.append("after")
    )
    env.restore_practice_scene(scene=env.backend().practice_scene())
    assert charged == []
    assert env.get_current_state().get(obj=env.scene, feature_name="steps_taken") == before


def test_each_evaluation_skill_is_preceded_by_the_evaluations_number_and_task() -> None:
    written: list[str] = []
    env, _ = _env(written=written)
    written.clear()
    env.backend().evaluation_progress(index=3, label="after cycle 3", task=2, num_tasks=5)
    env.backend().run_pick_cube()
    progress = written.index("eval 3 after cycle 3: task 2/5")
    assert written[progress + 1].startswith("running PickCube")
    written.clear()
    env.backend().evaluation_progress(index=4, label="(final)", task=1, num_tasks=5)
    env.backend().reset(seed=0)
    assert written[0] == "eval 4 (final): task 1/5"


def test_a_timed_human_reset_shows_the_seconds_so_far_while_waiting_for_done() -> None:
    """While the person resets, a counter ticks the seconds since their start y (Josh,
    2026-11-06), so they can see the time being charged. It stops at the done y."""
    ticks: list[str] = []
    seen_a_tick = threading.Event()
    now = iter([100.0, *(100.0 + k for k in range(1, 1000))])

    def read(prompt: str) -> str:  # noqa: PLR0917 -- `TerminalPerson.read`'s callable
        if "in place" in prompt:
            assert seen_a_tick.wait(timeout=5)
        return "y"

    def tick(text: str) -> None:  # noqa: PLR0917 -- `TerminalPerson.tick`'s callable
        ticks.append(text)
        seen_a_tick.set()

    person = TerminalPerson(
        write=lambda _line: None,
        read=read,
        time_human_resets=True,
        clock=lambda: next(now),
        tick=tick,
        tick_period_s=0.01,
    )
    request = PlacementRequest(
        kind="human reset",
        destination=Tossing3DSide.ROBOT,
        cube=CUBE_SPOT,
        bin=BIN_SPOTS[Tossing3DSide.ROBOT],
    )
    person.place(request=request)
    assert ticks and all(text.endswith(" s") for text in ticks)
    assert ticks[0] == "reset timer: 1 s"
    stopped = len(ticks)
    time.sleep(0.05)
    assert len(ticks) == stopped


def test_an_untimed_placement_shows_no_counter() -> None:
    ticks: list[str] = []
    person = TerminalPerson(
        write=lambda _line: None,
        read=lambda _prompt: "y",
        tick=ticks.append,
        tick_period_s=0.01,
    )
    request = PlacementRequest(
        kind="human reset",
        destination=Tossing3DSide.ROBOT,
        cube=CUBE_SPOT,
        bin=BIN_SPOTS[Tossing3DSide.ROBOT],
    )
    person.place(request=request)
    assert ticks == []


class _MisplacesTheBinOnce(FakePerson):
    """A FakePerson who first puts the bin `offset` (map frame) from its spot, and on
    the spot when asked to move it."""

    offset: tuple[float, float] = (0.0, 0.0)
    misplaced: int = 0

    def placed(self, *, request: PlacementRequest) -> None:
        if self.misplaced == 0:
            self.misplaced += 1
            spot = request.bin.center
            self.robot.put(
                cube=request.cube,
                bin_center=(spot[0] + self.offset[0], spot[1] + self.offset[1]),
                bin_yaw=request.bin.yaw,
            )
            return
        super().placed(request=request)


def test_a_bin_put_off_the_close_spot_is_accepted_there() -> None:
    """Only the far spot's placement matters (Josh, 2026-11-08)."""
    written: list[str] = []
    robot = FakeTossingRobot()
    person = _MisplacesTheBinOnce(robot=robot, write=written.append, offset=(0.0, 0.0))
    env = RealTossing3DEnvironment(tossing_robot=robot, person=person, clock=robot.now)
    env.hard_reset()
    person.offset, person.misplaced = (0.07, 0.21), 0
    written.clear()
    robot.calls.clear()
    UnconditionalHumanOracle.execute_movables_reset(env=env, destination="robot_side")
    assert not [line for line in written if line.startswith("the bin is")]
    assert [name for name, _ in robot.calls].count("observe") == 1


@pytest.mark.parametrize("destination", ["opposite_side"])
def test_a_bin_put_off_its_spot_is_reported_and_the_person_asked_to_move_it(
    *, destination: str
) -> None:
    """Trial 3 on 2026-11-08 started with the far bin 21 cm across the tape from its spot,
    and the first toss was refused; the run should say so before any skill instead."""
    written: list[str] = []
    robot = FakeTossingRobot()
    person = _MisplacesTheBinOnce(robot=robot, write=written.append, offset=(0.0, 0.0))
    env = RealTossing3DEnvironment(tossing_robot=robot, person=person, clock=robot.now)
    env.hard_reset()
    person.offset, person.misplaced = (0.07, 0.21), 0
    written.clear()
    robot.calls.clear()
    UnconditionalHumanOracle.execute_movables_reset(env=env, destination=destination)
    (problem,) = [line for line in written if line.startswith("the bin is")]
    assert "22 cm from its spot" in problem
    assert "Moved the bin onto its spot? [y/n]" in "\n".join(written)
    assert [name for name, _ in robot.calls].count("observe") == 2
    assert not MOTION & {name for name, _ in robot.calls}
    spot = BIN_SPOTS[Tossing3DSide(destination)].center
    assert robot.observe()["bin"]["x"] == pytest.approx(spot[0])
    assert robot.observe()["bin"]["y"] == pytest.approx(spot[1])


def test_a_bin_within_a_few_centimetres_of_its_spot_is_accepted_as_placed() -> None:
    written: list[str] = []
    robot = FakeTossingRobot()
    person = _MisplacesTheBinOnce(robot=robot, write=written.append, offset=(0.02, -0.02))
    env = RealTossing3DEnvironment(tossing_robot=robot, person=person, clock=robot.now)
    env.hard_reset()
    assert not [line for line in written if line.startswith("the bin is")]
    assert [name for name, _ in robot.calls].count("observe") == 1


def _fresh_look(*, robot: FakeTossingRobot) -> tuple[Tossing3DObservation, frozenset]:
    """What a look at the room now would give the planner: the observation and atoms."""
    backend = RealTossingBackend(
        robot=robot.model_copy(deep=True),
        person=FakePerson(robot=robot, write=lambda _line: None),
        parameters=REAL_TOSS_PARAMETERS,
    )
    return backend.observe(), backend.abstract_atoms()


def test_a_pick_that_holds_the_cube_skips_the_look_after_it() -> None:
    """A held pick already says what a look would (Josh, 2026-11-08): holding, the cube
    out of view, the robot at its stand, the gripper closed. So the next observation is
    built from the pick's reply, and the camera round after it is saved."""
    env, robot = _env()
    robot.calls.clear()
    env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0], dtype=float))
    assert robot.holding
    assert [name for name, _ in robot.calls] == ["pick"]
    observation, atoms = _fresh_look(robot=robot)
    assert env.backend().observe() == observation
    assert env.backend().abstract_atoms() == atoms
    assert "Holding" in {name for name, _ in atoms}
    assert env.get_current_state().get(obj=env.robot, feature_name="pos_gripper") == 1.0
    assert [name for name, _ in robot.calls] == ["pick"]


def test_the_skill_after_a_held_pick_looks_as_before() -> None:
    env, robot = _env()
    env.backend().run_pick_cube()
    robot.calls.clear()
    env.backend().run_open_gripper()
    assert [name for name, _ in robot.calls][:2] == ["open_gripper", "observe"]


class _MissingRobot(FakeTossingRobot):
    """A pick that moves and closes on nothing: a real miss, not a refusal."""

    def pick(self) -> dict[str, object]:
        self.calls.append(("pick", ()))
        return {
            "held": False,
            "gripper": 0.0,
            "note": "gripper closed on nothing",
            "refused": False,
            "reason": None,
            "cube": None,
            "stand": list(self.robot),
        }


def test_a_pick_that_misses_still_looks() -> None:
    robot = _MissingRobot()
    env = RealTossing3DEnvironment(
        tossing_robot=robot, person=FakePerson(robot=robot, write=lambda _line: None)
    )
    env.hard_reset()
    robot.calls.clear()
    run = env.backend().run_pick_cube()
    assert run.error is not None
    assert [name for name, _ in robot.calls][:2] == ["pick", "observe"]


def test_an_evaluation_drives_home_only_before_its_first_task() -> None:
    """Josh, 2026-11-08: a later task starts where the one before left the robot; the
    gripper is still opened before every task."""
    robot = FakeTossingRobot()
    evaluation = RealTossing3DEnvironment(
        tossing_robot=robot,
        person=FakePerson(robot=robot, write=lambda _line: None),
        evaluation=True,
    )
    backend = evaluation.backend()
    for task in range(1, 5):
        backend.evaluation_progress(index=2, label="after cycle 2", task=task, num_tasks=4)
        backend.reset(seed=0)
        backend.evaluation_progress(index=2, label="after cycle 2", task=task, num_tasks=4)
        backend.run_pick_cube()
    names = [name for name, _ in robot.calls]
    assert names.count("go_home") == 1
    assert names.count("open_gripper") == 4
    assert names.index("go_home") < names.index("open_gripper") < names.index("pick")


def test_practice_scenes_and_restores_still_drive_home_every_time() -> None:
    env, robot = _env()
    robot.calls.clear()
    env.backend().reset(seed=0)
    env.backend().reset(seed=0)
    env.restore_practice_scene(
        scene=PracticeScene(cube=CUBE_SPOT, holding=False, bin_side=Tossing3DSide.OPPOSITE)
    )
    names = [name for name, _ in robot.calls]
    assert names.count("go_home") == 3
    assert names.count("open_gripper") == 3


class _LosesTheCubeOnToss(FakeTossingRobot):
    """A toss after which no camera sees the cube (`lost` looks), then it is seen."""

    lost: int = 0

    def toss(
        self, *, standoff: float, speed: float, release_ms: float, direction: float
    ) -> dict[str, object]:
        reply = super().toss(
            standoff=standoff, speed=speed, release_ms=release_ms, direction=direction
        )
        self.cube = None
        return reply

    def observe(self) -> dict[str, object]:
        reply = super().observe()
        if self.lost > 0:
            self.lost -= 1
        elif self.cube is None:
            self.cube = (0.7, -0.6, 0.0)
            reply = super().observe()
            self.calls.pop()
        return reply


class _Answers(FakePerson):
    """A FakePerson who answers the lost-cube question from `answers`, in order."""

    answers: list[str]

    def model_post_init(self, __context: object) -> None:
        self.read = lambda prompt: self._answer(prompt=prompt)

    def _answer(self, *, prompt: str) -> str:
        answer = self.answers.pop(0) if prompt.startswith("Where is it?") else "y"
        self.write(f"{prompt}{answer}")
        return answer

    def locate_lost_cube(self) -> str:
        return TerminalPerson.locate_lost_cube(self)


def _lost_after_toss(*, answers: list[str]) -> tuple[Tossing3DObservation, list[str], int]:
    written: list[str] = []
    robot = _LosesTheCubeOnToss()
    person = _Answers(robot=robot, write=written.append, answers=answers)
    env = RealTossing3DEnvironment(tossing_robot=robot, person=person, clock=robot.now)
    env.hard_reset()
    env.backend().run_pick_cube()
    robot.calls.clear()
    written.clear()
    robot.lost = 1
    env.backend().run_move_to_toss_location_and_toss(
        distance=1.2, rotation=math.pi, release_speed_deg_s=103.0, gripper_release_ms=590.0
    )
    looks = [name for name, _ in robot.calls].count("observe")
    return env.backend().observe(), written, looks


def test_a_cube_no_camera_sees_after_a_skill_is_asked_about_and_in_the_bin_scores() -> None:
    """Josh, 2026-11-08 (the cardboard box): when the bottle is seen neither on the floor
    nor in the bin, ask where it is rather than take it as gone."""
    seen, written, looks = _lost_after_toss(answers=["b"])
    assert any("the cameras do not see the cube" in line for line in written)
    assert seen.solved
    assert looks == 1


def test_a_lost_cube_out_of_the_workspace_stays_unseen() -> None:
    seen, _written, looks = _lost_after_toss(answers=["o"])
    assert not seen.solved
    assert looks == 1


def test_look_again_looks_and_a_cube_then_seen_asks_no_more() -> None:
    seen, written, looks = _lost_after_toss(answers=["x", "l"])
    assert "'x': please type b, o, l or n." in written
    assert looks == 2
    assert not seen.solved
    assert sum(line.startswith("Where is it?") for line in written) == 2


def test_an_evaluations_progress_line_shows_its_score_so_far() -> None:
    """Josh, 2026-11-08: each task's line carries the robot's in-bin count of the tasks
    before it, so the score is visible during an evaluation."""
    robot = FakeTossingRobot()
    lines: list[str] = []
    evaluation = RealTossing3DEnvironment(
        tossing_robot=robot,
        person=FakePerson(robot=robot, write=lambda _line: None),
        evaluation=True,
    )
    backend = evaluation.backend()
    backend._say = lambda *, line: lines.append(line)  # type: ignore[method-assign]
    landed_in_bin = [True, False, True]
    for task, in_bin in enumerate(landed_in_bin, start=1):
        backend.evaluation_progress(index=1, label="after cycle 2", task=task, num_tasks=4)
        backend.reset(seed=0)
        if in_bin:
            robot.cube = (robot.bin[0], robot.bin[1], 0.0)
        backend._seen = None
        backend.observe()
    backend.evaluation_progress(index=1, label="after cycle 2", task=4, num_tasks=4)
    backend.reset(seed=0)
    progress = [line for line in lines if line.startswith("eval 1")]
    assert progress[0] == "eval 1 after cycle 2: task 1/4"
    assert progress[1] == "eval 1 after cycle 2: task 2/4 · 1/1 in the bin so far"
    assert progress[2] == "eval 1 after cycle 2: task 3/4 · 1/2 in the bin so far"
    assert progress[3] == "eval 1 after cycle 2: task 4/4 · 2/3 in the bin so far"


class _BinUnseenOnce(FakePerson):
    """A FakePerson whose first placement leaves no bin the cameras can see."""

    hidden: int = 0

    def placed(self, *, request: PlacementRequest) -> None:
        super().placed(request=request)
        if self.hidden == 0:
            self.hidden += 1
            self.robot.bin = None


def test_a_far_bin_the_cameras_do_not_see_is_reported_and_the_person_asked_to_place_it() -> None:
    """Trial 5's evaluation task 10 on 2026-11-08: the scene look found no box, and the
    task went on to no throw rather than asking for the bin."""
    written: list[str] = []
    robot = FakeTossingRobot()
    person = _BinUnseenOnce(robot=robot, write=written.append)
    env = RealTossing3DEnvironment(tossing_robot=robot, person=person, clock=robot.now)
    env.hard_reset()
    (problem,) = [line for line in written if line.startswith("the cameras see no bin")]
    assert "far spot" in problem
    assert "Moved the bin onto its spot? [y/n]" in "\n".join(written)
    assert [name for name, _ in robot.calls].count("observe") == 2
    spot = BIN_SPOTS[Tossing3DSide.OPPOSITE].center
    assert robot.observe()["bin"]["x"] == pytest.approx(spot[0])
