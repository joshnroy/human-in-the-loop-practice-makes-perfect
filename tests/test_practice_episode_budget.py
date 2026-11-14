"""`--practice-episode-budget`: practice counted in tries instead of control steps.

One episode is one completed execution of a skill with learnable parameters. The step
budget stays as a backstop. The fakes are `test_step_protocol.py`'s; a script item
`("try", n)` is a parameterised skill of `n` control steps, a bare int a setup skill."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.core.method.method import HumanCubeBinResetRequested
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.step_protocol import PracticeClock, StepPracticeRunner, StepProtocolValidation
from tests.environments.tworoutes.run_fixtures import TwoRoutesRun
from tests.test_step_protocol import (
    FakeEvaluations,
    FakeProblem,
    FakeStepProtocol,
    ScriptedMethod,
    _events,
    _step_args,
)


class ScriptedTries(ScriptedMethod):
    """Records what the loop tells the method and when, in one ordered list."""

    def model_post_init(self, context, /) -> None:  # noqa: PLR0917
        super().model_post_init(context)
        self._last_was_try = False
        self._calls: list = []
        self._asked_whether_episode = 0

    def get_practice_policy(self, *, task):
        def policy(state):  # noqa: PLR0917 -- Policy is a positional callable
            item = self._script.pop(0) if self._script else 1
            if isinstance(item, BaseException):
                raise item
            self._last_was_try = isinstance(item, tuple)
            steps = item[1] if isinstance(item, tuple) else item
            label = f"try{steps}" if self._last_was_try else f"setup{steps}"
            return SimpleNamespace(action=np.array([float(steps)]), label=label)

        return policy

    def practice_episode_dispatched(self) -> bool:
        self._asked_whether_episode += 1
        return self._last_was_try

    def observe_practice_action_budget(self, **budget) -> None:
        self._calls.append(("budget", budget))

    def observe_session_end(self, *, state) -> None:
        self._calls.append(("session_end", state))

    def end_cycle(self) -> None:
        self._calls.append(("end_cycle", None))


def _method(*, script: list) -> ScriptedTries:
    env = LightSwitchEnvironment(grid_size=4)
    method = ScriptedTries(env=env, skill_provider=LightSwitchSkillProvider(env=env), seed=0)
    method._script = list(script)
    return method


def _run(*, tmp_path, script: list, episodes: int | None, budget: int = 1000, **extra):
    output = tmp_path / "run"
    method = _method(script=script)
    if episodes is not None:
        extra["practice_episode_budget"] = episodes
    StepPracticeRunner.run(
        environment=FakeStepProtocol,
        args=_step_args(output_dir=output, budget=budget, interval=100_000, **extra),
        method=method,
        problem=FakeProblem(),
        evaluations=FakeEvaluations(),
    )
    return output, method


def _status(*, output) -> dict:
    return json.loads((output / "status.json").read_text())


def _end(*, output) -> dict:
    return next(e for e in _events(output=output) if e["event"] == "practice_end")


HUMAN = HumanCubeBinResetRequested(cost=1.0, destination="robot_side")


def test_practice_ends_after_exactly_n_tries_and_setup_and_human_actions_are_not_tries(*, tmp_path):
    script = [2, ("try", 3), HUMAN, ("try", 1), 4, 4, ("try", 2), HUMAN, ("try", 5)]
    output, method = _run(tmp_path=tmp_path, script=script, episodes=3)
    events = _events(output=output)
    robot = [e for e in events if e["event"] == "robot"]
    assert [e["label"] for e in robot] == ["setup2", "try3", "try1", "setup4", "setup4", "try2"]
    assert [e["practice_episodes"] for e in robot] == [0, 1, 2, 2, 2, 3]
    # The person's return after the last try is not waited for.
    assert sum(e["event"] == "human" for e in events) == 1
    end = _end(output=output)
    assert end["reason"] == "practice_episode_budget"
    assert end["practice_steps"] == 2 + 3 + 1 + 1 + 4 + 4 + 2
    assert end["practice_episodes"] == 3
    status = _status(output=output)
    assert status["practice_endpoint"] == "practice_episode_budget"
    assert status["practice_episodes"] == 3
    assert status["practice_episode_budget"] == 3
    assert status["practice_episode_budget_reached"] is True
    assert method._script == [HUMAN, ("try", 5)]
    protocol = json.loads((output / "step_protocol.json").read_text())
    assert protocol["practice_episode_budget"] == 3


def test_the_last_try_is_observed_and_learned_from_before_the_final_measurement(*, tmp_path):
    output, method = _run(tmp_path=tmp_path, script=[("try", 2), ("try", 3)], episodes=2)
    # The last try has no next policy call to be scored on: the session end scores it,
    # then the learning update runs, and only then is the policy measured.
    assert [name for name, _ in method._calls][-2:] == ["session_end", "end_cycle"]
    assert method._calls[-2][1] == ("after", 3.0)
    events = _events(output=output)
    names = [e["event"] for e in events]
    update = next(e for e in events if e["event"] == "learning_update")
    assert update["reason"] == "practice_episode_budget"
    final = max(i for i, name in enumerate(names) if name == "measurement")
    assert names.index("learning_update") < final
    records = json.loads((output / "measurements.json").read_text())
    assert records[-1]["learning_updates"] == 1
    assert records[-1]["practice_episodes"] == 2


def test_a_run_the_step_budget_ends_first_says_so(*, tmp_path):
    output, _ = _run(tmp_path=tmp_path, script=[("try", 2)] * 9, episodes=5, budget=6)
    assert _end(output=output)["reason"] == "practice_step_budget"
    status = _status(output=output)
    assert status["practice_endpoint"] == "practice_step_budget"
    assert status["practice_episodes"] == 3
    assert status["practice_episode_budget"] == 5
    assert status["practice_episode_budget_reached"] is False


def test_a_try_that_lands_on_both_budgets_at_once_ends_on_the_episode_budget(*, tmp_path):
    output, _ = _run(tmp_path=tmp_path, script=[("try", 2)] * 9, episodes=3, budget=6)
    assert _end(output=output)["reason"] == "practice_episode_budget"


def test_a_refused_try_is_an_episode_and_a_try_the_step_budget_cuts_is_not(*, tmp_path):
    # A refusal returns from the controller with no control step: an outcome the method
    # observes. A cut try raises out of it: no outcome exists.
    output, _ = _run(
        tmp_path=tmp_path, script=[("try", 0), ("try", 2), ("try", 9)], episodes=5, budget=5
    )
    robot = [e for e in _events(output=output) if e["event"] == "robot"]
    assert [(e["end_step"] - e["start_step"], e["interrupted"]) for e in robot] == [
        (0, False),
        (2, False),
        (3, True),
    ]
    assert [e["practice_episodes"] for e in robot] == [1, 2, 2]
    assert _status(output=output)["practice_episodes"] == 2


def test_the_method_is_told_the_tries_left_before_every_action(*, tmp_path):
    _, method = _run(tmp_path=tmp_path, script=[1, ("try", 1), HUMAN, ("try", 1)], episodes=2)
    told = [budget["remaining_episodes"] for name, budget in method._calls if name == "budget"]
    assert told == [2, 2, 1, 1]


def test_without_the_flag_nothing_about_episodes_is_asked_passed_or_written(*, tmp_path):
    script = [2, ("try", 3), HUMAN, ("try", 1)]
    output, method = _run(tmp_path=tmp_path, script=script, episodes=None, budget=9)
    assert method._asked_whether_episode == 0
    budgets = [budget for name, budget in method._calls if name == "budget"]
    assert budgets and all(set(b) == {"remaining_actions", "remaining_steps"} for b in budgets)
    for event in _events(output=output):
        assert not any("episode" in key for key in event), event
    for name in ("status.json", "step_protocol.json", "measurements.json"):
        assert "episode" not in (output / name).read_text(), name
    assert _end(output=output)["reason"] == "practice_step_budget"


def test_measurements_can_be_taken_every_k_tries(*, tmp_path):
    output, _ = _run(
        tmp_path=tmp_path,
        script=[("try", 2), 1, ("try", 2), ("try", 2), ("try", 2)],
        episodes=4,
        measurement_interval_episodes=2,
    )
    records = json.loads((output / "measurements.json").read_text())
    # Before practice, after the second try, and the final one (never twice at the end).
    assert [r["practice_episodes"] for r in records] == [0, 2, 4]
    assert [r["practice_steps"] for r in records] == [0, 5, 9]
    protocol = json.loads((output / "step_protocol.json").read_text())
    assert protocol["measurement_interval_episodes"] == 2


def test_the_tries_completed_are_part_of_the_clock_a_checkpoint_saves():
    clock = PracticeClock(budget=100, interval=10, episode_budget=5, on_measure=lambda: None)
    clock.robot_steps = 7
    clock.episodes = 3
    restored = PracticeClock(budget=100, interval=10, episode_budget=5, on_measure=lambda: None)
    for field, value in clock.checkpoint_fields().items():
        setattr(restored, field, value)
    assert restored.episodes == 3
    assert restored.steps == clock.steps
    # A run with no episode budget saves exactly what it saved before.
    plain = PracticeClock(budget=100, interval=10, on_measure=lambda: None)
    assert set(plain.checkpoint_fields()) == {
        "robot_steps",
        "stop_steps",
        "human_invocations",
        "counted_human_steps",
    }


def test_an_episode_budget_needs_the_step_budget_as_its_backstop(*, tmp_path):
    with pytest.raises(SystemExit):
        Cli.parse_args(
            argv=["--env", "lightswitch", "--method", "ees", "--practice-episode-budget", "5"]
        )
    args = _step_args(
        output_dir=tmp_path / "run", budget=10, interval=5, measurement_interval_episodes=2
    )
    with pytest.raises(ValueError, match="--practice-episode-budget"):
        StepProtocolValidation.check(args=args)


@pytest.mark.parametrize("flag", ["--practice-episode-budget", "--measurement-interval-episodes"])
@pytest.mark.parametrize("value", ["0", "-3"])
def test_episode_flags_reject_non_positive_values_at_parse_time(*, flag, value):
    with pytest.raises(SystemExit):
        Cli.parse_args(argv=["--env", "lightswitch", "--method", "ees", flag, value])


def test_a_method_that_cannot_say_what_a_try_is_refuses_an_episode_budget(*, tmp_path):
    from hitl_pmp.core.method.method import Method

    with pytest.raises(NotImplementedError, match="episode"):
        Method.practice_episode_dispatched(SimpleNamespace())


def test_two_routes_has_no_parameterised_skill_so_its_run_ends_on_the_backstop(*, tmp_path):
    """The real CLI, EES: every Two Routes skill has `param_dim == 0`."""
    output = TwoRoutesRun.run(
        directory=tmp_path, method="ees", budget=12, extra=("--practice-episode-budget", "3")
    )
    status = _status(output=output)
    assert status["practice_endpoint"] == "practice_step_budget"
    assert status["practice_episodes"] == 0
    assert status["practice_episode_budget_reached"] is False


def test_a_run_recorded_before_the_episode_flags_existed_still_resumes():
    from hitl_pmp.step_protocol import StepProtocolResume

    before = {"seed": "0", "env": "racktoserving3d"}
    now = {**before, "practice_episode_budget": "None", "measurement_interval_episodes": "None"}
    assert StepProtocolResume.comparable(arguments=before) == StepProtocolResume.comparable(
        arguments=now
    )
    assert StepProtocolResume.comparable(arguments=before) != StepProtocolResume.comparable(
        arguments={**now, "practice_episode_budget": "5"}
    )
