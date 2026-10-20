"""The real backend under the sim's own environment, predicates and toss pipeline."""

import logging
import math
import re
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.predicates import BIN_AT_SIDE, CUBE_AT_SIDE
from hitl_pmp.environments.tossing3d.real.backend import RealSkillStepCosts, TerminalPerson
from hitl_pmp.environments.tossing3d.real.environment import RealTossing3DEnvironment
from hitl_pmp.environments.tossing3d.real.fake_robot import FakePerson, FakeTossingRobot
from hitl_pmp.environments.tossing3d.real.frame import BIN_SPOTS, CUBE_SPOT, ROOM_FRAME
from hitl_pmp.environments.tossing3d.real.session_log import RealSessionLog
from hitl_pmp.environments.tossing3d.sides import Tossing3DSide, Tossing3DSides
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.environments.tossing3d.toss import Tossing3DToss
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import REAL_TOSS_PARAMETERS
from hitl_pmp.humans.oracle import UnconditionalHumanOracle

MOTION = {"pick", "toss", "open_gripper", "retract", "go_home"}


def _env(*, written: list[str] | None = None) -> tuple[RealTossing3DEnvironment, FakeTossingRobot]:
    robot = FakeTossingRobot()
    lines = written if written is not None else []
    person = FakePerson(robot=robot, write=lines.append)
    env = RealTossing3DEnvironment(tossing_robot=robot, person=person)
    env.hard_reset()
    return env, robot


def _side(*, env: RealTossing3DEnvironment, predicate: object, obj: object) -> set[str]:
    state = env.get_current_state()
    return {
        side.name
        for side in Tossing3DSides.objects()
        if predicate.holds(state=state, objects=(obj, env.barrier, side))  # type: ignore[attr-defined]
    }


def test_the_task_scene_is_the_cube_at_its_spot_and_the_bin_across_the_tape() -> None:
    written: list[str] = []
    env, _ = _env(written=written)
    state = env.get_current_state()
    cube_xy = ROOM_FRAME.to_domain(x=CUBE_SPOT[0], y=CUBE_SPOT[1])
    assert state.get(obj=env.cube, feature_name="x") == pytest.approx(cube_xy[0])
    assert state.get(obj=env.barrier, feature_name="x") == pytest.approx(0.0, abs=1e-3)
    assert state.get(obj=env.barrier, feature_name="bb_x") == pytest.approx(0.05)
    assert _side(env=env, predicate=CUBE_AT_SIDE, obj=env.cube) == {"robot_side"}
    assert _side(env=env, predicate=BIN_AT_SIDE, obj=env.bin) == {"opposite_side"}
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
    assert _side(env=env, predicate=BIN_AT_SIDE, obj=env.bin) == {expected_side}
    assert _side(env=env, predicate=CUBE_AT_SIDE, obj=env.cube) == {"robot_side"}


def _toss_skill(*, env: RealTossing3DEnvironment, side: str) -> GroundSkill:
    return GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier, Tossing3DSides.parse(name=side)),
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
    skill = _toss_skill(env=env, side=destination)
    bounds = Tossing3DToss.standoff_bounds(
        ground_skill=skill, state=state, parameters=REAL_TOSS_PARAMETERS
    )
    assert bounds == REAL_TOSS_PARAMETERS.standoff_bounds
    params = np.array([1.42, 120.0, 675.0])
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
    assert (standoff, speed, release_ms) == (1.42, 120.0, 675.0)
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
        action=np.array([env.move_to_toss_location_and_toss_id, 1.4, -math.pi / 2, 120, 675])
    )
    assert env.last_controller_steps() == (0,)
    assert "refused" in (env.last_skill_error() or "")
    assert robot.holding


def test_each_real_skill_is_charged_its_sim_mean_ticks_after_one_before_check() -> None:
    env, _ = _env()
    counts = {"before": 0, "after": 0}

    def before() -> None:
        counts["before"] += 1

    def after() -> None:
        counts["after"] += 1

    env.set_control_step_observers(before=before, after=after)
    env.take_action(action=np.array([env.pick_cube_id, 0, 0, 0, 0], dtype=float))
    assert counts == {"before": 1, "after": RealSkillStepCosts().pick_cube}
    assert RealSkillStepCosts() == RealSkillStepCosts(
        pick_cube=119, move_to_toss_location_and_toss=90, open_gripper=1
    )


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


def test_a_blank_answer_asks_again_instead_of_ending_the_run() -> None:
    robot = FakeTossingRobot()
    person = _Asked(robot=robot, write=lambda _line: None, answers=["", "  ", "y"], asked=[])
    env = RealTossing3DEnvironment(tossing_robot=robot, person=person, confirm_each_skill=True)
    env.hard_reset()
    env.backend().run_pick_cube()
    assert len(person.asked) == 3
    assert robot.observe()["holding"]


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
    env.backend().run_pick_cube()
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
    declines = iter(["", "No"])
    person = TerminalPerson(write=lines.append, read=lambda _prompt: next(declines))
    with pytest.raises(KeyboardInterrupt):
        person.confirm(action="PickCube")


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
