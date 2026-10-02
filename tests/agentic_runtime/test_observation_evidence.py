"""Only bounded chronological physical measurements become judgment history."""

import json
from pathlib import Path
from typing import Any

import pytest

from hitl_pmp.agentic_runtime.observations import ObservationEvidence


class EvidenceFixtures:
    @staticmethod
    def observation(*, step: int) -> dict[str, Any]:
        return {
            "observation_mode": "object_state",
            "objects": [{"name": "cube", "type": "cube", "features": {"z": step / 100}}],
            "proprioception": {"pos_tool_z": 0.03 + step / 100},
            "control_step": step,
            "state_spec": {"length_unit": "m"},
        }


def test_history_keeps_initial_middle_and_last_three_physical_samples(*, tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    with path.open("w") as stream:
        for step in range(265):
            observation = EvidenceFixtures.observation(step=step)
            observation["execution_seed"] = 7
            observation["execution"] = {"controller_done": True}
            observation["recent_observations"] = [{"success": True}]
            stream.write(
                json.dumps({
                    "operation": "step",
                    "action": [0.1],
                    "error": "Not physical evidence",
                    "done": True,
                    "observation": observation,
                })
                + "\n"
            )
        stream.write(json.dumps({"operation": "finish", "done": True, "error": None}) + "\n")
    current = EvidenceFixtures.observation(step=264)
    result = ObservationEvidence.with_history(observation=current, trajectory_path=str(path))
    samples = result["recent_observations"]
    steps = [sample["control_step"] for sample in samples]
    assert len(samples) == 8
    assert steps == sorted(set(steps))
    assert steps[0] == 0 and steps[-3:] == [262, 263, 264]
    assert any(80 <= step <= 180 for step in steps)
    assert all(
        sample == EvidenceFixtures.observation(step=step)
        for sample, step in zip(samples, steps, strict=True)
    )
    assert "recent_observations" not in current


def test_before_is_preserved_and_small_trajectory_keeps_every_physical_frame(
    *, tmp_path: Path
) -> None:
    path = tmp_path / "trajectory.jsonl"
    frames = [EvidenceFixtures.observation(step=step) for step in (1, 2, 3)]
    path.write_text(
        "\n".join(json.dumps({"operation": "observe", "observation": frame}) for frame in frames)
    )
    before = EvidenceFixtures.observation(step=0)
    result = ObservationEvidence.with_history(
        observation=frames[-1], trajectory_path=str(path), before=before
    )
    assert result["recent_observations"] == [before, *frames]
    result["recent_observations"][0]["objects"][0]["features"]["z"] = 999
    assert before["objects"][0]["features"]["z"] == 0


def test_no_trajectory_uses_only_before_and_does_not_invent_history() -> None:
    current = EvidenceFixtures.observation(step=3)
    before = EvidenceFixtures.observation(step=0)
    assert ObservationEvidence.with_history(observation=current) == current
    result = ObservationEvidence.with_history(observation=current, before=before)
    assert result["recent_observations"] == [before]
    assert result["objects"] == current["objects"]


def test_finish_annotations_alone_are_not_observation_evidence(*, tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    path.write_text('{"operation":"finish","done":true,"success":true}\n')
    current = EvidenceFixtures.observation(step=0)
    assert (
        ObservationEvidence.with_history(observation=current, trajectory_path=str(path)) == current
    )


def test_missing_host_trajectory_is_not_silently_ignored(*, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ObservationEvidence.with_history(
            observation=EvidenceFixtures.observation(step=0),
            trajectory_path=str(tmp_path / "missing.jsonl"),
        )


def test_rgb_is_unchanged_and_does_not_read_numeric_history(*, tmp_path: Path) -> None:
    rgb = {"observation_mode": "rgb", "images": ["data:image/png;base64,YQ=="]}
    assert (
        ObservationEvidence.with_history(
            observation=rgb,
            trajectory_path=str(tmp_path / "missing.jsonl"),
            before=EvidenceFixtures.observation(step=0),
        )
        == rgb
    )


def test_malformed_host_trajectory_is_reported(*, tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    path.write_text('{"observation":\n')
    with pytest.raises(ValueError):
        ObservationEvidence.with_history(
            observation=EvidenceFixtures.observation(step=0), trajectory_path=str(path)
        )
