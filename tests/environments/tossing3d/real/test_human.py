"""The real human reset: a person is asked, confirms, and the scene is re-observed."""

import json

import pytest

from hitl_pmp.core.problem.human.human import HumanEndedSession
from hitl_pmp.environments.tossing3d.predicates import BIN_AT_SIDE, CUBE_AT_SIDE
from hitl_pmp.environments.tossing3d.real.backend import RealWorld
from hitl_pmp.environments.tossing3d.real.environment import RealTossing3DEnvironment
from hitl_pmp.environments.tossing3d.real.fake_robot import FakeRobotPlacer, FakeTossingRobot
from hitl_pmp.environments.tossing3d.real.human import RealHumanOracle, TerminalScenePlacer
from hitl_pmp.environments.tossing3d.real.types import PlacementRequest, PlanarRegion
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.humans.oracle import UnconditionalHumanOracle


class ScriptedPerson:
    """Answers prompts from a script and records what it was shown."""

    def __init__(self, *, answers: list[str]) -> None:
        self.answers = list(answers)
        self.shown: list[str] = []
        self.asked: list[str] = []

    def input(self, prompt: str) -> str:  # noqa: PLR0917 - stands in for builtins.input
        self.asked.append(prompt)
        return self.answers.pop(0)

    def output(self, text: str) -> None:  # noqa: PLR0917 - stands in for builtins.print
        self.shown.append(text)


class FirstAttemptAcrossTheBarrier(FakeRobotPlacer):
    """A person who, the first time, leaves the cube across the barrier."""

    def place(self, *, request: PlacementRequest) -> None:
        super().place(request=request)
        if request.kind == "reset" and len(self.requests()) == 2:
            self.robot.place(
                cube=self.robot.cube_start.model_copy(update={"x": -0.5}),
                bin_=self.robot._bin,  # noqa: SLF001 - the fake's own world
            )


def _env(*, placer_type: type[FakeRobotPlacer] = FakeRobotPlacer, tmp_path=None):
    robot = FakeTossingRobot()
    world = RealWorld(
        robot=robot,
        placer=placer_type(robot=robot),
        event_log=None if tmp_path is None else tmp_path / "events.jsonl",
    )
    env = RealTossing3DEnvironment(world=world)
    env.reset_to_seed(seed=1)
    return env, robot, world


def test_the_terminal_placer_says_what_to_do_and_waits_for_enter() -> None:
    person = ScriptedPerson(answers=[""])
    placer = TerminalScenePlacer(input_fn=person.input, output_fn=person.output)
    placer.place(
        request=PlacementRequest(
            kind="reset",
            owner="practice",
            cube_target=(0.8, 0.1),
            cube_region=PlanarRegion(x_min=0.6, y_min=-0.4, x_max=1.0, y_max=0.4),
            bin_target=(-0.8, 0.0),
            destination="opposite_side",
        )
    )
    text = "\n".join(person.shown)
    assert "cube" in text and "(0.80, 0.10)" in text
    assert "bin" in text and "(-0.80, 0.00)" in text
    assert "far side" in text
    assert len(person.asked) == 1 and "Enter" in person.asked[0]


def test_a_restore_of_a_held_cube_says_it_cannot_be_put_back_in_the_gripper() -> None:
    person = ScriptedPerson(answers=[""])
    TerminalScenePlacer(input_fn=person.input, output_fn=person.output).place(
        request=PlacementRequest(
            kind="restore",
            owner="practice",
            cube_target=(0.8, 0.1),
            bin_target=(-0.8, 0.0),
            was_holding=True,
        )
    )
    assert "gripper" in "\n".join(person.shown)


def test_notes_from_a_failed_check_are_shown() -> None:
    person = ScriptedPerson(answers=[""])
    TerminalScenePlacer(input_fn=person.input, output_fn=person.output).place(
        request=PlacementRequest(
            kind="reset",
            owner="practice",
            cube_target=(0.8, 0.1),
            bin_target=(-0.8, 0.0),
            notes=("the cube is not on the robot's side",),
        )
    )
    assert "the cube is not on the robot's side" in "\n".join(person.shown)


def test_the_oracle_prices_commands_as_the_sims_does() -> None:
    assert RealHumanOracle.intervention_cost == UnconditionalHumanOracle.intervention_cost


def test_a_reset_puts_the_bin_on_the_requested_side(*, tmp_path) -> None:
    env, robot, world = _env(tmp_path=tmp_path)
    RealHumanOracle.execute_movables_reset(env=env, destination="robot_side")
    state = env.get_current_state()
    assert BIN_AT_SIDE.holds(state, (env.bin, env.barrier, Tossing3DSides.robot))
    assert CUBE_AT_SIDE.holds(state, (env.cube, env.barrier, Tossing3DSides.robot))
    record = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "human_reset"
    ][-1]
    assert record["destination"] == "robot_side"
    assert record["attempts"] == 1
    assert record["unmet"] == []
    assert record["seconds"] >= 0
    del robot, world


def test_a_reset_that_did_not_take_is_asked_for_again_with_the_reason() -> None:
    env, robot, world = _env(placer_type=FirstAttemptAcrossTheBarrier)
    RealHumanOracle.execute_movables_reset(env=env, destination="opposite_side")
    requests = world.placer.requests()
    resets = [request for request in requests if request.kind == "reset"]
    assert len(resets) == 2
    assert any("robot's side" in note for note in resets[1].notes)
    assert CUBE_AT_SIDE.holds(
        env.get_current_state(), (env.cube, env.barrier, Tossing3DSides.robot)
    )
    event = [e for e in world.events() if e["event"] == "human_reset"][-1]
    assert event["attempts"] == 2
    del robot


def test_a_reset_that_never_takes_stops_asking_and_says_so() -> None:
    class AlwaysAcross(FakeRobotPlacer):
        def place(self, *, request: PlacementRequest) -> None:
            super().place(request=request)
            if request.kind == "reset":
                self.robot.place(
                    cube=self.robot.cube_start.model_copy(update={"x": -0.5}),
                    bin_=self.robot._bin,  # noqa: SLF001
                )

    env, _, world = _env(placer_type=AlwaysAcross)
    RealHumanOracle.execute_movables_reset(env=env, destination="opposite_side")
    event = [e for e in world.events() if e["event"] == "human_reset"][-1]
    assert event["attempts"] == RealHumanOracle.max_attempts
    assert event["unmet"]
    # The fake person was asked, and its scripted answer is to carry on.
    assert world.placer.unverified_asks() == 1


def test_a_command_with_a_target_state_sets_up_that_scene() -> None:
    env, _, world = _env()
    initial = env.get_current_state()
    RealHumanOracle.execute_human_command(
        command_start_state_description=None,
        command_goal_description=type("Goal", (), {"target_state": initial})(),
        env=env,
    )
    assert world.placer.requests()[-1].kind == "scene"
    assert world.placer.requests()[-1].seed == 1


def test_the_oracle_refuses_a_sim_environment() -> None:
    from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment

    with pytest.raises(TypeError, match="RealTossing3DEnvironment"):
        RealHumanOracle.execute_movables_reset(env=Tossing3DEnvironment(), destination=None)


def test_the_terminal_placer_asks_whether_a_hidden_cube_is_in_the_bin() -> None:
    person = ScriptedPerson(answers=["maybe", "y"])
    placer = TerminalScenePlacer(input_fn=person.input, output_fn=person.output)
    assert placer.inspect_cube_in_bin(owner="practice") is True
    assert len(person.asked) == 2
    assert (
        TerminalScenePlacer(
            input_fn=ScriptedPerson(answers=["n"]).input, output_fn=person.output
        ).inspect_cube_in_bin(owner="practice")
        is False
    )


# --- a reset check that fails every attempt: ask, do not carry on silently ----------


class AlwaysAcrossAtTheTerminal(TerminalScenePlacer):
    """A person at the terminal who, every time, leaves the cube across the barrier.

    Every prompt -- Enter after a placement, and the question after the last failed
    check -- goes through the one scripted `input_fn`, as at the real terminal.
    """

    robot: FakeTossingRobot

    def place(self, *, request: PlacementRequest) -> None:
        super().place(request=request)
        FakeRobotPlacer(robot=self.robot).place(request=request)
        if request.kind == "reset":
            self.robot.place(
                cube=self.robot.cube_start.model_copy(update={"x": -0.5}),
                bin_=self.robot._bin,  # noqa: SLF001 - the fake's own world
            )


def _terminal_env(*, answers: list[str]):
    person = ScriptedPerson(answers=answers)
    robot = FakeTossingRobot()
    placer = AlwaysAcrossAtTheTerminal(robot=robot, input_fn=person.input, output_fn=person.output)
    world = RealWorld(robot=robot, placer=placer)
    env = RealTossing3DEnvironment(world=world)
    env.reset_to_seed(seed=1)
    return env, world, person


def test_the_terminal_asks_whether_to_continue_after_an_unverified_reset() -> None:
    person = ScriptedPerson(answers=["maybe", "c"])
    placer = TerminalScenePlacer(input_fn=person.input, output_fn=person.output)
    assert placer.continue_after_unverified_reset(
        owner="practice", unmet=("the cube is not on the robot's side",), attempts=3
    )
    assert len(person.asked) == 2
    assert "the cube is not on the robot's side" in "\n".join(person.shown)
    person = ScriptedPerson(answers=["e"])
    placer = TerminalScenePlacer(input_fn=person.input, output_fn=person.output)
    assert not placer.continue_after_unverified_reset(
        owner="practice", unmet=("the bin is not seen upright",), attempts=3
    )


def test_three_failed_checks_pause_and_continue_on_the_persons_word() -> None:
    # Enter for the scene and each of the three placements, then "c": continue anyway.
    env, world, person = _terminal_env(answers=["", "", "", "", "c"])
    RealHumanOracle.execute_movables_reset(env=env, destination="opposite_side")
    assert len(person.asked) == 1 + RealHumanOracle.max_attempts + 1
    assert "continue" in person.asked[-1].lower()
    event = [e for e in world.events() if e["event"] == "human_reset"][-1]
    assert event["attempts"] == RealHumanOracle.max_attempts
    assert event["unmet"]
    assert event["after_unverified"] == "continue"


def test_three_failed_checks_pause_and_end_the_session_on_the_persons_word() -> None:
    env, world, person = _terminal_env(answers=["", "", "", "", "e"])
    with pytest.raises(HumanEndedSession):
        RealHumanOracle.execute_movables_reset(env=env, destination="opposite_side")
    assert len(person.asked) == 1 + RealHumanOracle.max_attempts + 1
    event = [e for e in world.events() if e["event"] == "human_reset"][-1]
    assert event["after_unverified"] == "end"


def test_a_reset_that_takes_asks_nothing_more() -> None:
    env, _, world = _env()
    RealHumanOracle.execute_movables_reset(env=env, destination="opposite_side")
    assert world.placer.unverified_asks() == 0
    event = [e for e in world.events() if e["event"] == "human_reset"][-1]
    assert event["after_unverified"] is None


def test_the_fake_placer_answers_the_unverified_reset_question_as_scripted() -> None:
    robot = FakeTossingRobot()
    assert FakeRobotPlacer(robot=robot).continue_after_unverified_reset(
        owner="practice", unmet=("x",), attempts=3
    )
    placer = FakeRobotPlacer(robot=robot, continue_after_unverified=False)
    assert not placer.continue_after_unverified_reset(owner="practice", unmet=("x",), attempts=3)
    assert placer.unverified_asks() == 1
