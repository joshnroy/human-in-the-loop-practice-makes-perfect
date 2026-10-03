"""CLI composition preserves the initialized world across generation and practice."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from hitl_pmp.agentic_runtime.sandbox import PolicyExecution
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.environments.tossing3d.agentic_bridge import AgenticTossing3DEnvironment
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.problem import Tossing3DProblem
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks
from hitl_pmp.methods.agentic_options import cli
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    TOSS_SKILL,
)
from hitl_pmp.practice_loop import PracticeResetPolicy

from .support import task
from .test_cli import complete_library, input_bundle


class LifecycleFixture:
    """All costly boundaries are replaced; the CLI and method composition stay real."""

    def __init__(
        self, *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_at: str | None = None
    ) -> None:
        self.events: list[str] = []
        self.reset_envs: list[AgenticTossing3DEnvironment] = []
        self.closed_envs: list[AgenticTossing3DEnvironment] = []
        self.bridges: list[SimpleNamespace] = []
        self.observation = {
            "observation_mode": "object_state",
            "objects": [{"name": "cube_0", "type": "cube", "features": {"x": 0.37}}],
            "proprioception": {"pos_arm_joint1": 0.12},
            "control_step": 0,
            "action_spec": {"shape": [18]},
        }
        self.robot_spec = {"robot_name": "actual_spawned_robot", "kinematic_chain": {}}
        self.human = {
            "HUMAN_TASK_QUESTION": "Describe desired tasks and important state features.",
            "HUMAN_TASK_RESPONSE": "Toss the cube into the bin; pay attention to the barrier.",
        }
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        bundle = {
            **input_bundle(),
            "observation_mode": "object_state",
            "evaluation_skill_order": [TOSS_SKILL, PICK_SKILL, OPEN_GRIPPER_SKILL],
            "task": "task.md",
            "robot_api": "robot_api.md",
            "human_input": "intake.json",
            "prompts": {
                name: f"{name}.md" for name in ("generate", "classify", "verify_option", "improve")
            },
        }
        (inputs / "bundle.json").write_text(json.dumps(bundle))
        (inputs / "intake.json").write_text(json.dumps(self.human))
        for name in [bundle["task"], bundle["robot_api"], *bundle["prompts"].values()]:
            (inputs / name).write_text(f"External fixture {name}")
        runtime_path = tmp_path / "runtime.json"
        runtime_path.write_text(
            json.dumps({
                "coding": {
                    "sandbox": {
                        "image": "fixture-image",
                        "robocode_checkout": str(tmp_path / "robocode"),
                        "artifact_dir": str(tmp_path / "unused"),
                    },
                    "artifact_dir": str(tmp_path / "unused"),
                    "backend": "codex",
                    "model": "fixture-model",
                    "max_budget_usd": 1,
                },
                "vision": {"provider": "robocode_broker"},
            })
        )
        self.args = argparse.Namespace(
            env="tossing3d",
            output_dir=tmp_path / "output",
            practice_reset_policy=PracticeResetPolicy.NEVER,
            practice_reset_interval=None,
            agentic_inputs=inputs,
            agentic_runtime_config=runtime_path,
            agentic_library=None,
            seed=125,
            human_reset=True,
            human_reset_practice_cost=5.0,
            agentic_search_depth=2,
            agentic_num_particles=8,
            agentic_linear_cost_lambda=None,
            num_cycles=1,
            max_steps_per_interaction=2,
        )

        def record(*, event: str) -> None:
            self.events.append(event)
            if fail_at == event:
                raise RuntimeError(f"fixture failure at {event}")

        def hard_reset(env: AgenticTossing3DEnvironment) -> None:  # noqa: PLR0917
            self.reset_envs.append(env)
            record(event="reset")
            env.current_state = State(data={})

        def close(env: AgenticTossing3DEnvironment) -> None:  # noqa: PLR0917
            self.closed_envs.append(env)

        def make_bridge(
            *, env: AgenticTossing3DEnvironment, observation_mode: str, step_limit: int = 1000
        ) -> SimpleNamespace:
            def observe() -> dict[str, Any]:
                assert env.current_state is not None
                record(event="observe")
                return self.observation

            def robot_spec() -> dict[str, Any]:
                assert env.current_state is not None
                record(event="robot_spec")
                return self.robot_spec

            bridge = SimpleNamespace(
                env=env,
                observation_mode=observation_mode,
                step_limit=step_limit,
                observe=observe,
                robot_spec=robot_spec,
            )
            self.bridges.append(bridge)
            return bridge

        def generate(*, prompt: str):
            assert "External fixture generate.md" in prompt
            assert self.events == ["reset", "observe", "robot_spec"]
            record(event="generate")
            return complete_library()

        self.agent = Mock()
        self.agent.generate.side_effect = generate
        self.agent_factory = Mock(return_value=self.agent)
        self.executor = Mock()
        self.executor.execute.return_value = PolicyExecution(
            steps=1,
            controller_done=True,
            final_observation=self.observation,
            trajectory_path=str(tmp_path / "fixture.jsonl"),
        )
        self.runner = Mock()
        monkeypatch.setattr(cli, "RobocodeCodingAgent", self.agent_factory)
        monkeypatch.setattr(cli, "Tossing3DAgenticBridge", make_bridge)
        monkeypatch.setattr(cli, "DockerPolicyExecutor", Mock(return_value=self.executor))
        monkeypatch.setattr(cli, "RobocodeVisionClient", Mock())
        monkeypatch.setattr(cli.DockerContainer, "image_id", lambda _: "sha256:fixture")
        monkeypatch.setattr(cli.MethodRunner, "run", self.runner)
        monkeypatch.setattr(AgenticTossing3DEnvironment, "hard_reset", hard_reset)
        monkeypatch.setattr(AgenticTossing3DEnvironment, "close", close)
        self.built_problems = []
        for name in ("build_practice_problem", "build_evaluation_problem"):
            env = Tossing3DEnvironment(scene_bg=False)
            problem = Tossing3DProblem(env=env, tasks=Tossing3DTasks(env=env))
            self.built_problems.append(problem)
            monkeypatch.setattr(cli.Tossing3DCli, name, Mock(return_value=problem))

    def run(self) -> None:
        cli.AgenticOptionsCli.run(args=self.args, env_cli=cli.Tossing3DCli)


def test_spawned_world_and_grounding_inputs_reach_generation_then_same_practice(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = LifecycleFixture(tmp_path=tmp_path, monkeypatch=monkeypatch)

    def enter_practice(**kwargs: Any) -> None:
        printed = capsys.readouterr().out
        assert fixture.human["HUMAN_TASK_RESPONSE"] in printed
        assert "Generated skills and abstract states" in printed
        assert complete_library().manifest.revision in printed
        for cluster in complete_library().manifest.clusters:
            assert cluster.description in printed
        for option in complete_library().manifest.options:
            assert option.option_id in printed and option.initiation in printed
        method = kwargs["method"]
        assert kwargs["practice_initialized"] is True
        assert kwargs["problem"].env is fixture.reset_envs[0] is method.env
        assert kwargs["evaluation_problem"].env is method.evaluation_env
        assert kwargs["problem"].tasks.env is method.env
        assert kwargs["evaluation_problem"].tasks.env is method.evaluation_env
        assert method.practice_observer.env is method.env
        assert method.evaluation_observer.env is method.evaluation_env
        assert method.evaluation_skill_order == (TOSS_SKILL, PICK_SKILL, OPEN_GRIPPER_SKILL)
        assert method.practice_observer.observation_mode == "object_state"
        assert method.evaluation_observer.observation_mode == "object_state"
        for env in (method.env, method.evaluation_env):
            assert len(env._policy_executors) == 3
            for executor in env._policy_executors.values():
                assert executor.observation_mode == "object_state"
                if env is method.evaluation_env:
                    assert executor.last_execution is method.evaluation_execution
                else:
                    assert executor.last_execution is None
        evaluation_journal = method.evaluation_execution
        evaluation_journal["trajectory_path"] = "previous-task-history.jsonl"
        method.execution_records["practice-only"] = {"trajectory_path": "practice-history.jsonl"}
        method.get_task_policy(task=task())
        assert method.evaluation_execution is evaluation_journal
        assert evaluation_journal == {}
        assert (
            method.execution_records["practice-only"]["trajectory_path"] == "practice-history.jsonl"
        )
        method.env._policy_executors[method.action_ids["pick"]](params=np.zeros(4))
        executed_bridge = fixture.executor.execute.call_args.kwargs["bridge"]
        assert executed_bridge.env is method.env
        assert executed_bridge.observation_mode == "object_state"
        fixture.events.append("practice")

    fixture.runner.side_effect = enter_practice
    fixture.run()
    assert fixture.events == ["reset", "observe", "robot_spec", "generate", "practice"]
    assert len(fixture.reset_envs) == 1
    assert len(fixture.closed_envs) == 2
    assert fixture.closed_envs[0] is fixture.reset_envs[0]
    assert fixture.closed_envs[0] is not fixture.closed_envs[1]
    staged = fixture.agent_factory.call_args.kwargs["input_files"]
    expected = {
        "human_input.json": fixture.human,
        "initial_observation.json": fixture.observation,
        "robot_spec.json": fixture.robot_spec,
    }
    assert set(staged) == set(expected)
    for name, value in expected.items():
        assert json.loads(staged[name]) == value
        assert json.loads((fixture.args.output_dir / name).read_text()) == value
    assert (fixture.args.output_dir / "initial_library.json").is_file()


@pytest.mark.parametrize("fail_at", ["reset", "observe", "generate"])
def test_worlds_close_if_initialization_or_generation_fails(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_at: str
) -> None:
    fixture = LifecycleFixture(tmp_path=tmp_path, monkeypatch=monkeypatch, fail_at=fail_at)
    with pytest.raises(RuntimeError, match=f"fixture failure at {fail_at}"):
        fixture.run()
    fixture.runner.assert_not_called()
    assert len(fixture.closed_envs) == 2
    assert fixture.closed_envs[0] is fixture.reset_envs[0]
    assert fixture.closed_envs[0] is not fixture.closed_envs[1]
