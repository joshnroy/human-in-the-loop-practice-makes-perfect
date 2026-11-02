"""Terminal evaluation failures stop control and policy retries, without physics."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.cli_protocols import SimulatedStepProtocolHooks
from hitl_pmp.core.control_steps import ControlStepLimitReached
from hitl_pmp.core.method.types import LabeledAction
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Goal, Task
from hitl_pmp.environments.tossing3d.real.step_protocol import Tossing3DRealStepProtocol
from hitl_pmp.recording.run_record.types import RecordingHeader
from hitl_pmp.recording.run_record.writer import RunRecordWriter
from hitl_pmp.step_protocol import DeploymentSnapshot, EvaluationClock, StepEvaluation


@pytest.mark.parametrize("goal", [False, True])
def test_clock_latches_failure_ahead_of_goal_and_budget(*, goal):
    signal = {"failed": False}
    clock = EvaluationClock(
        budget=1, check_goal=lambda: goal, check_failure=lambda: signal["failed"]
    )
    clock.before()
    signal["failed"] = True
    clock.after()
    signal["failed"] = False
    with pytest.raises(ControlStepLimitReached):
        clock.before()
    assert clock.failed and not clock.solved
    assert clock.steps == 1
    assert clock.interruption() == "stopped: task failed"


@pytest.mark.parametrize("hooks", [SimulatedStepProtocolHooks, Tossing3DRealStepProtocol])
def test_existing_simulated_and_real_hooks_default_to_no_failure(*, hooks):
    assert not hooks.failure_check(problem=object(), task=object())()


@pytest.mark.parametrize("when", ["initial", "control_step", "skill_end"])
@pytest.mark.parametrize("goal_also_true", [False, True])
def test_evaluation_failure_stops_before_policy_can_recover_and_is_recorded(
    *, tmp_path, monkeypatch, when, goal_also_true
):
    state = State(data={})
    task = Task(initial_state=state, goal=Goal(atoms=frozenset()))
    signal = {"failed": when == "initial", "goal": when == "initial" and goal_also_true}
    calls = []

    class Environment:
        before = None
        after = None

        def set_control_step_observers(self, *, before, after):
            self.before, self.after = before, after

        def noop_action(self):
            return np.array([-1.0])

        def close(self):
            calls.append("close")

    env = Environment()

    def take_action(*, action):
        del action
        calls.append("action")
        env.before()
        if when == "control_step":
            signal.update(failed=True, goal=goal_also_true)
        env.after()
        if when == "control_step":
            env.before()
        signal.update(failed=True, goal=goal_also_true)
        return state

    problem = SimpleNamespace(
        env=env,
        hard_reset=lambda: None,
        sample_test_task=lambda: task,
        reset_to_task=lambda **kwargs: state,
        take_action=take_action,
    )

    def policy(*, state):
        del state
        calls.append("policy")
        if calls.count("policy") > 1:
            signal.update(failed=False, goal=True)
            return LabeledAction(action=env.noop_action(), label="rescue")
        return LabeledAction(action=np.array([1.0]), label="SweepTable")

    method = SimpleNamespace(get_task_policy=lambda **kwargs: lambda state: policy(state=state))
    monkeypatch.setattr(DeploymentSnapshot, "restore", lambda **kwargs: method)

    class Hooks(SimulatedStepProtocolHooks):
        build_evaluation_problem = staticmethod(lambda **kwargs: problem)
        evaluation_skill_provider = staticmethod(lambda **kwargs: object())
        task_seed = staticmethod(lambda **kwargs: 0)
        goal_check = staticmethod(lambda **kwargs: lambda: signal["goal"])
        failure_check = staticmethod(lambda **kwargs: lambda: signal["failed"])
        interrupted = staticmethod(lambda **kwargs: calls.append("interrupted"))
        last_skill_error = staticmethod(lambda **kwargs: None)
        last_skill_trace = staticmethod(lambda **kwargs: {"skill": "SweepTable"})

        @staticmethod
        def attach_recording(*, output_path, **kwargs):
            del kwargs
            return RunRecordWriter(
                output_path=output_path, header=RecordingHeader(environment="sweeptable3d")
            )

    snapshot = tmp_path / "snapshot.pickle"
    snapshot.write_bytes(b"mocked snapshot")
    output = tmp_path / "evaluation"
    result = StepEvaluation.run(
        snapshot=str(snapshot),
        configuration={"num_test_tasks": 1, "evaluation_control_steps": 5},
        output=str(output),
        environment=Hooks,
    )
    expected_actions = 0 if when == "initial" else 1
    assert calls.count("policy") == calls.count("action") == expected_actions
    assert calls[-1] == "close"
    assert env.before is None and env.after is None
    assert result["num_solved"] == 0
    assert result["tasks"][0]["reason"] == "task_failed"
    assert result["tasks"][0]["steps"] == expected_actions
    recording = [json.loads(line) for line in (output / "recording.jsonl").read_text().splitlines()]
    assert recording[-1]["outcome"] == "task_failed"
    if expected_actions:
        actions = [json.loads(line) for line in (output / "actions.jsonl").read_text().splitlines()]
        assert len(actions) == 1
        assert "failed" in actions[0]["outcome"]
        assert actions[0]["skill_trace"] == {"skill": "SweepTable"}
