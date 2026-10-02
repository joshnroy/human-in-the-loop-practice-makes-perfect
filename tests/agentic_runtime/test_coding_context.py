"""Generation and revision receive reproducible context and actual execution evidence."""

import json
from pathlib import Path
from typing import Any

import pytest

from hitl_pmp.agentic_runtime.robocode import CodingSettings, RobocodeCodingAgent
from hitl_pmp.agentic_runtime.sandbox import SandboxSettings
from hitl_pmp.agentic_runtime.types import GeneratedLibrary


class ContextFixtures:
    @staticmethod
    def settings(*, root: Path) -> CodingSettings:
        return CodingSettings(
            sandbox=SandboxSettings(
                image="offline-fixture", robocode_checkout=root, artifact_dir=root / "execution"
            ),
            artifact_dir=root / "coding",
            backend="codex",
            model="offline-fixture",
            max_budget_usd=1,
        )

    @staticmethod
    def library() -> GeneratedLibrary:
        return GeneratedLibrary.model_validate({
            "manifest": {
                "revision": "initial",
                "clusters": [{"cluster_id": "ready", "description": "The cube is within reach."}],
                "options": [
                    {
                        "option_id": "pick",
                        "belief_skill": "pick",
                        "initiation": "The cube is within reach.",
                        "termination": "The robot stops.",
                        "success": "The cube is lifted.",
                        "controller": "policies/pick.py",
                        "cost": 1,
                    }
                ],
                "edges": [],
            },
            "controllers": {
                "policies/pick.py": (
                    "def policy(observation, memory, parameters):\n    return {'done': True}\n"
                )
            },
        })

    @staticmethod
    def inputs() -> dict[str, str]:
        return {
            "human_response.txt": "Practice picking up the cube and tossing it into the bin.",
            "initial_observation.json": json.dumps({
                "observation_mode": "object_state",
                "objects": [{"name": "cube0", "type": "cube", "features": {"x": 0.4, "z": 0.03}}],
            }),
            "robot_spec.json": json.dumps({
                "joint_order": ["shoulder", "elbow"],
                "joint_unit": "rad",
            }),
        }


def test_generation_and_revision_share_copied_context_and_stage_real_trajectory(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = ContextFixtures.library()
    inputs = ContextFixtures.inputs()
    expected_inputs = dict(inputs)
    agent = RobocodeCodingAgent(
        settings=ContextFixtures.settings(root=tmp_path), input_files=inputs
    )
    # The caller's subsequent edits must not change the recorded experiment inputs.
    inputs["human_response.txt"] = "A different experiment."
    inputs["unrequested.txt"] = "Do not stage this."
    trajectory = tmp_path / "actual-execution.jsonl"
    trajectory_text = (
        '{"step":0,"action":[0.2],"observation":{"pos_arm_joint1":0.0}}\n'
        '{"step":1,"action":[0.1],"observation":{"pos_arm_joint1":0.1}}\n'
    )
    trajectory.write_text(trajectory_text)
    evidence = [{"trajectory_path": str(trajectory), "success": False, "resolved": True}]
    calls: list[dict[str, Any]] = []
    revised = library.controllers["policies/pick.py"] + "# correction from observed execution\n"

    def run(*, prompt: str, files: dict[str, str], output_filename: str) -> str:
        calls.append({"prompt": prompt, "files": dict(files), "output_filename": output_filename})
        # A launcher receives a fresh mapping, not ownership of the retained inputs.
        files["human_response.txt"] = "Launcher-local edit."
        if output_filename == "generated_library.json":
            return library.model_dump_json()
        return json.dumps({"controllers": {"policies/pick.py": revised}})

    monkeypatch.setattr(agent, "_run", run)
    assert agent.generate(prompt="Construct the requested skills.") == library
    result = agent.revise(library=library, evidence=evidence, prompt="Improve the practiced skill.")
    assert result.controllers == {"policies/pick.py": revised}
    generation, revision = calls
    assert generation["files"] == expected_inputs
    assert generation["prompt"] == "Construct the requested skills."
    assert revision["prompt"] == "Improve the practiced skill."
    for name, content in expected_inputs.items():
        assert revision["files"][name] == content
    assert "unrequested.txt" not in revision["files"]
    assert revision["files"]["policies/pick.py"] == library.controllers["policies/pick.py"]
    assert json.loads(revision["files"]["library.json"]) == library.model_dump(mode="json")
    staged_evidence = json.loads(revision["files"]["evidence.json"])
    assert staged_evidence == [
        {"trajectory_path": "trajectories/000000.jsonl", "success": False, "resolved": True}
    ]
    assert revision["files"][staged_evidence[0]["trajectory_path"]] == trajectory_text
    assert evidence[0]["trajectory_path"] == str(trajectory)
    assert trajectory.read_text() == trajectory_text


@pytest.mark.parametrize(
    "name",
    [
        "../escape.txt",
        "context/../../escape.txt",
        "/tmp/escape.txt",
        "generated_library.json",
        "revision.json",
        "",
        ".",
    ],
)
def test_context_cannot_escape_the_sandbox(*, tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="context paths"):
        RobocodeCodingAgent(
            settings=ContextFixtures.settings(root=tmp_path), input_files={name: "bad"}
        )


@pytest.mark.parametrize(
    "name", ["policies/pick.py", "library.json", "evidence.json", "trajectories/000000.jsonl"]
)
def test_context_cannot_replace_policy_library_or_evidence(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    calls: list[Any] = []
    trajectory = tmp_path / "execution.jsonl"
    trajectory.write_text('{"step":0}\n')

    def run(**kwargs: Any) -> str:
        calls.append(kwargs)
        return '{"controllers":{}}'

    with pytest.raises(ValueError):
        agent = RobocodeCodingAgent(
            settings=ContextFixtures.settings(root=tmp_path), input_files={name: "replace"}
        )
        monkeypatch.setattr(agent, "_run", run)
        agent.revise(
            library=ContextFixtures.library(),
            evidence=[{"trajectory_path": str(trajectory)}],
            prompt="improve",
        )
    assert not calls


@pytest.mark.parametrize("name", ["./library.json", "policies//pick.py", "policies/./pick.py"])
def test_alias_paths_cannot_bypass_context_collision_check(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    calls: list[Any] = []

    def run(**kwargs: Any) -> str:
        calls.append(kwargs)
        return '{"controllers":{}}'

    with pytest.raises(ValueError):
        agent = RobocodeCodingAgent(
            settings=ContextFixtures.settings(root=tmp_path), input_files={name: "replace"}
        )
        monkeypatch.setattr(agent, "_run", run)
        agent.revise(library=ContextFixtures.library(), evidence=[], prompt="improve")
    assert not calls
