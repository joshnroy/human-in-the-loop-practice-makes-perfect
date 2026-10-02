"""Behavioral checks for abstraction uncertainty, session isolation and robot access."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from hitl_pmp.agentic_runtime.learner import CodePolicyLearner
from hitl_pmp.agentic_runtime.robocode import RobocodeCodingAgent
from hitl_pmp.agentic_runtime.sandbox import (
    DockerContainer,
    DockerPolicyExecutor,
    RobotRelay,
    SandboxSettings,
)
from hitl_pmp.agentic_runtime.types import GeneratedLibrary, RevisionProposal, RuntimeManifest
from hitl_pmp.agentic_runtime.vision import VLMJudge


class Fixtures:
    @staticmethod
    def library() -> GeneratedLibrary:
        return GeneratedLibrary.model_validate({
            "manifest": {
                "revision": "fixture-1",
                "clusters": [
                    {"cluster_id": "ready", "description": "Object reachable"},
                    {"cluster_id": "blocked", "description": "Object unreachable"},
                ],
                "options": [
                    {
                        "option_id": "act",
                        "belief_skill": "Act",
                        "initiation": "Object reachable",
                        "termination": "Robot stopped",
                        "success": "Object moved",
                        "controller": "policies/act.py",
                        "cost": 1,
                    }
                ],
                "edges": [
                    {
                        "source": "ready",
                        "option_id": "act",
                        "outcome": "success",
                        "destination": "blocked",
                        "probability": 1,
                    },
                    {
                        "source": "ready",
                        "option_id": "act",
                        "outcome": "failure",
                        "destination": "ready",
                        "probability": 1,
                    },
                ],
            },
            "controllers": {
                "policies/act.py": (
                    "def policy(observation, memory, parameters):\n"
                    "    return {'action': [0.0], 'memory': {}, 'done': True}\n"
                )
            },
        })


class FakeVision:
    def __init__(self, *, result: str) -> None:
        self.result = result
        self.messages: list[dict[str, Any]] = []

    def complete(self, *, messages: list[dict[str, Any]]) -> str:
        self.messages = messages
        return self.result


class RevisingAgent:
    def __init__(self) -> None:
        self.calls = 0

    def revise(
        self, *, library: GeneratedLibrary, evidence: list[dict[str, Any]], prompt: str
    ) -> RevisionProposal:
        self.calls += 1
        assert evidence == [{"success": False}]
        return RevisionProposal(
            controllers={
                "policies/act.py": library.controllers["policies/act.py"]
                + "# revised control law\n"
            }
        )


class FakeBridge:
    def __init__(self) -> None:
        self.actions: list[Any] = []

    def observe(self) -> dict[str, Any]:
        return {"position": len(self.actions)}

    def step(self, *, action: dict[str, Any] | list[Any]) -> dict[str, Any]:
        self.actions.append(action)
        return self.observe()

    def action_spec(self) -> dict[str, Any]:
        return {"size": 1}


class TestManifest:
    @pytest.mark.parametrize(
        "path", ["../escape.py", "/tmp/escape.py", ".hidden.py", "x//y.py", "x\\y.py", "wrong.json"]
    )
    def test_reject_generated_path_escape(self, *, path: str) -> None:
        data = Fixtures.library().model_dump()
        data["manifest"]["options"][0]["controller"] = path
        data["controllers"] = {path: "pass"}
        with pytest.raises(ValidationError, match="safe relative Python"):
            GeneratedLibrary.model_validate(data)

    def test_require_failure_destination_and_normalized_probabilities(self) -> None:
        data = Fixtures.library().manifest.model_dump()
        data["edges"] = [data["edges"][0]]
        with pytest.raises(ValidationError, match="both success and failure"):
            RuntimeManifest.model_validate(data)
        data = Fixtures.library().manifest.model_dump()
        data["edges"][0]["probability"] = 0.5
        with pytest.raises(ValidationError, match="sum to one"):
            RuntimeManifest.model_validate(data)

    def test_human_target_is_not_a_free_form_transition(self) -> None:
        data = Fixtures.library().manifest.model_dump()
        data["options"][0]["controller"] = None
        data["options"][0]["human_destination"] = "ready"
        with pytest.raises(ValidationError, match="Human edges"):
            RuntimeManifest.model_validate(data)


class TestVLM:
    def test_forward_actual_images_and_preserve_unknown(self) -> None:
        client = FakeVision(result='{"cluster_id":null,"reason":"Occluded"}')
        judge = VLMJudge(
            client=client, classification_prompt="classify", option_check_prompt="check"
        )
        observation = {"images": ["data:image/png;base64,YQ=="], "proprioception": {"joint": 0.2}}
        result = judge.classify(observation=observation, manifest=Fixtures.library().manifest)
        assert result.cluster_id is None
        assert client.messages[1]["content"][1]["image_url"]["url"] == observation["images"][0]
        assert "proprioception" in client.messages[1]["content"][0]["text"]

    def test_reject_unknown_ids_and_missing_visual_observation(self) -> None:
        client = FakeVision(result='{"cluster_id":"invented","reason":"guess"}')
        judge = VLMJudge(
            client=client, classification_prompt="classify", option_check_prompt="check"
        )
        with pytest.raises(ValueError, match="unknown cluster"):
            judge.classify(
                observation={"images": ["data:image/png;base64,YQ=="]},
                manifest=Fixtures.library().manifest,
            )
        with pytest.raises(ValueError, match="actual image"):
            judge.classify(
                observation={"privileged_predicates": ["Holding"]},
                manifest=Fixtures.library().manifest,
            )

    def test_boolean_strings_are_not_success(self) -> None:
        client = FakeVision(result='{"value":"false","reason":"Uncertain"}')
        judge = VLMJudge(
            client=client, classification_prompt="classify", option_check_prompt="check"
        )
        with pytest.raises(ValidationError):
            judge.check(
                observation={"images": ["data:image/png;base64,YQ=="]},
                option=Fixtures.library().manifest.options[0],
                phase="success",
            )


class TestLearner:
    def test_revision_only_at_boundary_and_snapshot_cannot_modify_live_code(
        self, *, tmp_path: Path
    ) -> None:
        library = Fixtures.library()
        agent = RevisingAgent()
        learner = CodePolicyLearner(
            library=library, agent=agent, improvement_prompt="improve", artifact_dir=tmp_path
        )
        snapshot = learner.library
        snapshot.controllers["policies/act.py"] = "malicious alteration"
        assert learner.library == library
        with pytest.raises(RuntimeError, match="active practice"):
            learner.record(evidence={"success": False})
        learner.begin_session()
        learner.record(evidence={"success": False})
        assert agent.calls == 0 and learner.library == library
        revision = learner.end_session()
        assert agent.calls == 1
        assert revision.changed_files == ("policies/act.py",)
        assert revision.previous_digest != revision.digest
        assert learner.library.manifest == library.manifest
        assert (tmp_path / "session-0000/evidence.json").is_file()
        with pytest.raises(RuntimeError, match="No active"):
            learner.end_session()

    def test_no_evidence_does_not_call_coding_agent(self, *, tmp_path: Path) -> None:
        agent = RevisingAgent()
        learner = CodePolicyLearner(
            library=Fixtures.library(),
            agent=agent,
            improvement_prompt="improve",
            artifact_dir=tmp_path,
        )
        learner.begin_session()
        revision = learner.end_session()
        assert agent.calls == 0
        assert revision.previous_digest == revision.digest


class TestSandboxBoundary:
    def test_generated_name_cannot_replace_supervisor_and_error_preserves_world(
        self, *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library_data = Fixtures.library().model_dump()
        library_data["manifest"]["options"][0]["controller"] = "transport.py"
        library_data["controllers"] = {"transport.py": "raise RuntimeError('generated code')"}
        library = GeneratedLibrary.model_validate(library_data)
        supervisor = tmp_path / "src/robocode/utils/isolated_transport.py"
        supervisor.parent.mkdir(parents=True)
        trusted_source = "def verify_namespace(): pass\ndef verify_strict_runtime(): pass\n"
        supervisor.write_text(trusted_source)
        executor = DockerPolicyExecutor(
            settings=SandboxSettings(
                image="strict:test",
                robocode_checkout=tmp_path,
                artifact_dir=tmp_path / "artifacts",
                seed=7,
            )
        )

        def simulated_child(*, work: Path, relay: RobotRelay) -> None:
            assert (work / "transport.py").read_text() == trusted_source
            assert (work / "controllers/transport.py").read_text() == library.controllers[
                "transport.py"
            ]
            assert json.loads((work / "request.json").read_text())["seed"] == 7
            relay.dispatch(request={"operation": "observe"})
            relay.dispatch(request={"operation": "step", "action": [1]})
            relay.dispatch(
                request={
                    "operation": "finish",
                    "done": False,
                    "error": "ValueError: invalid next action",
                }
            )

        monkeypatch.setattr(executor, "_run", simulated_child)
        bridge = FakeBridge()
        result = executor.execute(
            library=library, option=library.manifest.options[0], bridge=bridge
        )
        assert result.steps == 1 and result.final_observation == {"position": 1}
        assert not result.controller_done
        assert result.error == "ValueError: invalid next action"
        assert result.seed == 7 and Path(result.trajectory_path).is_file()

    def test_trajectory_keeps_all_numeric_steps_but_only_endpoint_images(
        self, *, tmp_path: Path
    ) -> None:
        trace = tmp_path / "trajectory.jsonl"
        relay = RobotRelay(
            path=tmp_path / "socket", bridge=FakeBridge(), max_steps=5, trajectory_path=trace
        )
        try:
            for step in range(5):
                relay._record(
                    event={"operation": "step", "observation": {"images": ["png"], "joint": step}}
                )
            relay.record_final(observation={"images": ["final"], "joint": 5}, error=None)
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            assert [record["observation"]["joint"] for record in records] == list(range(6))
            assert sum("images" in record["observation"] for record in records) == 2
        finally:
            relay.server_close()

    def test_no_reset_and_no_actions_after_budget_or_finish(self, *, tmp_path: Path) -> None:
        bridge = FakeBridge()
        trace = tmp_path / "trajectory.jsonl"
        relay = RobotRelay(
            path=tmp_path / "robot.sock", bridge=bridge, max_steps=1, trajectory_path=trace
        )
        try:
            for operation in ("reset", "set_state", "human_reset", "render_source"):
                with pytest.raises(ValueError, match="Only observe"):
                    relay.dispatch(request={"operation": operation})
            assert relay.dispatch(request={"operation": "step", "action": [1]}) == {"position": 1}
            with pytest.raises(ValueError, match="budget"):
                relay.dispatch(request={"operation": "step", "action": [2]})
            relay.dispatch(request={"operation": "finish", "done": True})
            with pytest.raises(ValueError, match="already ended"):
                relay.dispatch(request={"operation": "step", "action": [3]})
            assert bridge.actions == [[1]]
            assert json.loads(trace.read_text().splitlines()[0])["action"] == [1]
        finally:
            relay.server_close()

    def test_launch_has_only_disconnected_namespace_and_readonly_task_mount(
        self, *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "hitl_pmp.agentic_runtime.sandbox.shutil.which", lambda _: "/usr/bin/docker"
        )
        monkeypatch.setattr(DockerContainer, "image_id", lambda self: "sha256:" + "a" * 64)
        executor = DockerPolicyExecutor(
            settings=SandboxSettings(image="strict:test", robocode_checkout=tmp_path)
        )
        command = executor.command(work_dir=tmp_path, name="test-only")
        assert command[command.index("--network") + 1] == "none"
        assert command[command.index("--cap-drop") + 1] == "ALL"
        assert command[command.index("--security-opt") + 1] == "no-new-privileges"
        assert command[command.index("--user") + 1] != "0:0"
        assert "--read-only" in command and "--pids-limit" in command
        assert "--privileged" not in command and "--cap-add" not in command
        assert command.count("--mount") == 1
        assert command[command.index("--mount") + 1].endswith("dst=/sandbox,readonly")
        assert "sha256:" + "a" * 64 in command

    def test_missing_docker_fails_instead_of_host_exec(
        self, *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("hitl_pmp.agentic_runtime.sandbox.shutil.which", lambda _: None)
        executor = DockerPolicyExecutor(
            settings=SandboxSettings(image="missing:test", robocode_checkout=tmp_path)
        )
        with pytest.raises(RuntimeError, match="never run on the host"):
            executor.command(work_dir=tmp_path, name="test-only")

    def test_coding_client_has_only_inert_broker_credentials(self) -> None:
        command, environment = RobocodeCodingAgent.model_client(
            backend="codex", command=["codex", "exec", "-"]
        )
        assert command[-1] == "-"
        assert 'model_provider="robocode"' in command
        assert 'model_providers.robocode.base_url="http://127.0.0.1:18080/v1"' in command
        assert environment["ROBOCODE_MODEL_TOKEN"] == "local-broker-no-provider-secret"
        assert "OPENAI_API_KEY" not in environment and "CODEX_API_KEY" not in environment
        assert environment["UV_OFFLINE"] == "1"
