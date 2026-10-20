"""The step protocol runs on any domain that implements `StepProtocolEnvironment`.

The domain here is deliberately not Tossing3D: Light Switch's dynamics, with every
action charged a fixed number of control steps and two human skills named like the
shelf domain's, so nothing in the runner can lean on a Tossing3D name or backend.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import PrivateAttr

from hitl_pmp.core.method.method import HumanCubeBinResetRequested
from hitl_pmp.core.method.skill_provider import DomainContext, SkillProvider
from hitl_pmp.core.method.types import Policy
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.lightswitch.cli import LightSwitchCli
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.problem import LightSwitchProblem
from hitl_pmp.environments.lightswitch.skill_provider import (
    LightSwitchOracle,
    LightSwitchSkillProvider,
)
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.humans.oracle import UnconditionalHumanOracle
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.practice_loop import PracticeResetPolicy
from hitl_pmp.recording.run_record.types import RecordingHeader
from hitl_pmp.recording.run_record.writer import RunRecordWriter
from hitl_pmp.step_protocol import StepPracticeRunner, StepProtocolEntry

CONTROL_STEPS_PER_ACTION = 400
# Light Switch's own `Goal.describe()`; the runner must not supply a string of its own.
GOAL = "LightOn(light)"


class CountingLightSwitch(LightSwitchEnvironment):
    """Light Switch whose every action runs a fixed number of observed control steps."""

    control_steps_per_action: ClassVar[int] = CONTROL_STEPS_PER_ACTION
    _before: Callable[[], None] | None = PrivateAttr(default=None)
    _after: Callable[[], None] | None = PrivateAttr(default=None)
    _human_resets: list[str] = PrivateAttr(default_factory=list)

    def set_control_step_observers(
        self, *, before: Callable[[], None] | None, after: Callable[[], None] | None
    ) -> None:
        self._before = before
        self._after = after

    def take_action(self, *, action: Action) -> State:
        state = super().take_action(action=action)
        for _ in range(self.control_steps_per_action):
            if self._before is not None:
                self._before()
            if self._after is not None:
                self._after()
        return state

    def reset_movables(self, *, destination: str | None = None) -> bool:
        assert destination is not None
        self._human_resets.append(destination)
        return True


class FakeStepProtocol:
    """A static-method container, like every `StepProtocolEnvironment`."""

    @staticmethod
    def build_practice_problem(*, args: argparse.Namespace) -> LightSwitchProblem:
        env = CountingLightSwitch(grid_size=4)
        tasks = LightSwitchTasks(env=env, seed=args.seed)
        return LightSwitchProblem(env=env, tasks=tasks, human=UnconditionalHumanOracle)

    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> LightSwitchProblem:
        return FakeStepProtocol.build_practice_problem(args=args)

    @staticmethod
    def practice_context(*, args: argparse.Namespace, problem: Any) -> DomainContext:
        del args
        return DomainContext(
            env=problem.env,
            skill_provider=LightSwitchSkillProvider(env=problem.env),
            oracle=LightSwitchOracle(env=problem.env),
        )

    @staticmethod
    def evaluation_pool() -> None:
        return None

    @staticmethod
    def evaluation_skill_provider(*, problem: Any) -> SkillProvider:
        return LightSwitchSkillProvider(env=problem.env)

    @staticmethod
    def attach_recording(
        *, args: argparse.Namespace, problem: Any, output_path: Path
    ) -> RunRecordWriter:
        # No state source: the recording still carries every action and outcome.
        del problem
        return RunRecordWriter(
            output_path=output_path,
            header=RecordingHeader(environment="lightswitch", method=args.method),
        )

    @staticmethod
    def human_skill_names() -> dict[str, str]:
        return {"reset_cubes": "reset_cubes", "reset_wiper": "reset_wiper"}

    @staticmethod
    def task_seed(*, task: Task) -> int:
        return int(
            round(
                1000
                * task.initial_state.get(obj=LightSwitchEnvironment.light, feature_name="target")
            )
        )

    @staticmethod
    def goal_check(*, problem: Any, task: Task) -> Callable[[], bool]:
        return lambda: task.goal.is_satisfied(state=problem.env.get_current_state())

    @staticmethod
    def interrupted(*, problem: Any, action: Action) -> None:
        del problem, action

    @staticmethod
    def adopt_interrupted_skill(*, problem: Any, action: Action, reason: str) -> Any:
        del action, reason
        return problem.env.get_current_state()

    @staticmethod
    def last_skill_error(*, problem: Any) -> str | None:
        del problem
        return None

    @staticmethod
    def last_skill_trace(*, problem: Any) -> dict[str, Any] | None:
        del problem
        return {"skill": "fake"}

    progress: ClassVar[list[dict[str, int]]] = []

    @staticmethod
    def practice_progress(
        *, problem: Any, cycle: int, action: int, max_actions: int, steps: int, budget: int
    ) -> None:
        del problem
        FakeStepProtocol.progress.append({
            "cycle": cycle,
            "action": action,
            "max_actions": max_actions,
            "steps": steps,
            "budget": budget,
        })


class AsksForTheWiperOnce(EesMethod):
    """EES that opens practice by asking the human for `reset_wiper`."""

    _asked: bool = PrivateAttr(default=False)

    def get_practice_policy(self, *, task: Task) -> Policy:
        inner = super().get_practice_policy(task=task)

        def policy(state: State) -> Any:  # noqa: PLR0917 -- the Policy interface
            if not self._asked:
                self._asked = True
                raise HumanCubeBinResetRequested(cost=2.0, destination="reset_wiper")
            return inner(state)

        return policy


def _args(*, output: Path, **overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = dict(
        env="fake-counting-lightswitch",
        method="ees",
        seed=0,
        num_test_tasks=2,
        num_cycles=50,
        max_steps_per_interaction=20,
        practice_reset_policy=PracticeResetPolicy.NEVER,
        practice_reset_interval=None,
        output_dir=output,
        defer_rendering=True,
        practice_step_budget=15000 + CONTROL_STEPS_PER_ACTION,
        measurement_interval_steps=15000,
        evaluation_control_steps=3000,
        human_skill_steps=1,
        stop_after_perfect_evaluations=0,
        practice_cost_config=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class AsksForTheWiperOnceCli:
    """A method CLI, so the run goes through `StepProtocolEntry` like `cli.py`'s does."""

    built: ClassVar[list[AsksForTheWiperOnce]] = []
    domains: ClassVar[list[str]] = []

    @staticmethod
    def run(*, args: argparse.Namespace, env_cli: Any) -> None:
        # Method CLIs that only support some domains dispatch on this name.
        AsksForTheWiperOnceCli.domains.append(env_cli.__name__)

        def factory(context: DomainContext) -> AsksForTheWiperOnce:  # noqa: PLR0917
            method = AsksForTheWiperOnce(
                env=context.env, skill_provider=context.skill_provider, seed=args.seed
            )
            AsksForTheWiperOnceCli.built.append(method)
            return method

        env_cli.run_method(
            args=args,
            method_factory=factory,
            num_cycles=args.num_cycles,
            max_steps_per_interaction=args.max_steps_per_interaction,
        )


def _run(*, args: argparse.Namespace) -> tuple[CountingLightSwitch, Path]:
    AsksForTheWiperOnceCli.built.clear()
    AsksForTheWiperOnceCli.domains.clear()
    StepProtocolEntry.run(
        args=args,
        method_cli=AsksForTheWiperOnceCli,
        env_cli=LightSwitchCli,
        environment=FakeStepProtocol,
    )
    (method,) = AsksForTheWiperOnceCli.built
    assert AsksForTheWiperOnceCli.domains == ["LightSwitchCli"]
    assert isinstance(method.env, CountingLightSwitch)
    return method.env, Path(args.output_dir)


def test_a_non_tossing_domain_runs_the_step_protocol_end_to_end(*, tmp_path: Path) -> None:
    env, output = _run(args=_args(output=tmp_path / "run"))

    protocol = json.loads((output / "step_protocol.json").read_text())
    assert protocol["evaluation_control_steps"] == 3000
    assert protocol["measurement_interval_steps"] == 15000

    events = [json.loads(line) for line in (output / "step_events.jsonl").read_text().splitlines()]
    humans = [e for e in events if e["event"] == "human"]
    assert [(h["destination"], h["skill"]) for h in humans] == [("reset_wiper", "reset_wiper")]
    assert env._human_resets == ["reset_wiper"]
    robots = [e for e in events if e["event"] == "robot"]
    assert robots and all(r["skill_error"] is None for r in robots)
    assert all(r["end_step"] - r["start_step"] <= CONTROL_STEPS_PER_ACTION for r in robots)

    status = json.loads((output / "status.json").read_text())
    assert status["phase"] == "complete"
    assert status["human_resets_by_destination"] == {"reset_cubes": 0, "reset_wiper": 1}

    evaluations = json.loads((output / "step_evaluations.json").read_text())
    assert evaluations and all(e["complete"] for e in evaluations)
    assert all(e["num_total"] == 2 for e in evaluations)
    for evaluation in evaluations:
        for task in evaluation["tasks"]:
            assert task["steps"] <= 3000
            # Evaluation stops at the first control step the live goal check passes.
            assert task["solved"] == (task["reason"] == "goal")
            assert task["goal"] == GOAL
    stats = json.loads((output / "stats.json").read_text())
    goals = {
        outcome["goal"] for breakdown in stats["breakdowns"] for outcome in breakdown["outcomes"]
    }
    assert goals == {GOAL}
    for path in (output / "recording.jsonl", output / "evaluations" / "0000" / "recording.jsonl"):
        kinds = [json.loads(line)["kind"] for line in path.read_text().splitlines()]
        assert kinds[0] == "header"
        assert "action" in kinds
        assert kinds[-1] == "segment_end"


def test_every_robot_skill_logs_the_domains_skill_trace(*, tmp_path: Path) -> None:
    _, output = _run(args=_args(output=tmp_path / "run"))
    events = [json.loads(line) for line in (output / "step_events.jsonl").read_text().splitlines()]
    robots = [e for e in events if e["event"] == "robot"]
    assert robots and all(r["skill_trace"]["skill"] == "fake" for r in robots)
    others = [e for e in events if e["event"] != "robot"]
    assert all("skill_trace" not in e for e in others)
    actions = [
        json.loads(line)
        for path in sorted((output / "evaluations").glob("*/actions.jsonl"))
        for line in path.read_text().splitlines()
    ]
    assert actions and all(a["skill_trace"]["skill"] == "fake" for a in actions)
    assert all("skill_error" in a for a in actions)


def test_an_unknown_human_destination_is_refused(*, tmp_path: Path) -> None:
    class AsksForSomethingElse(AsksForTheWiperOnce):
        def get_practice_policy(self, *, task: Task) -> Policy:
            def policy(state: State) -> Any:  # noqa: PLR0917 -- the Policy interface
                raise HumanCubeBinResetRequested(cost=1.0, destination="robot_side")

            return policy

    args = _args(output=tmp_path / "run")
    problem = FakeStepProtocol.build_practice_problem(args=args)
    method = AsksForSomethingElse(
        env=problem.env, skill_provider=LightSwitchSkillProvider(env=problem.env)
    )
    with pytest.raises(ValueError, match="robot_side"):
        StepPracticeRunner.run(
            args=args, method=method, problem=problem, environment=FakeStepProtocol
        )


def test_a_domain_without_control_steps_is_refused():
    env = LightSwitchEnvironment(grid_size=4)
    with pytest.raises(NotImplementedError, match="control steps"):
        env.set_control_step_observers(before=lambda: None, after=lambda: None)


def test_the_domain_is_told_each_actions_place_in_the_cycle_and_the_budget(
    *, tmp_path: Path
) -> None:
    FakeStepProtocol.progress.clear()
    _, output = _run(args=_args(output=tmp_path / "run"))
    events = [json.loads(line) for line in (output / "step_events.jsonl").read_text().splitlines()]
    actions = [e for e in events if e["event"] in {"robot", "human", "stop"}]
    progress = FakeStepProtocol.progress
    # One call before each action, carrying the clock as that action starts.
    assert len(progress) >= len(actions) > 0
    for call, event in zip(progress, actions, strict=False):
        assert call["cycle"] == event["cycle"]
        assert call["steps"] == event["start_step"]
    budget = 15000 + CONTROL_STEPS_PER_ACTION
    assert all(c["budget"] == budget and c["max_actions"] == 20 for c in progress)
    first_cycle = [c["action"] for c in progress if c["cycle"] == 0]
    assert first_cycle == list(range(1, len(first_cycle) + 1))
