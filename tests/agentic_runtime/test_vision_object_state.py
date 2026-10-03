"""Numeric language judgments use declared measurements, never an oracle outcome."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from hitl_pmp.agentic_runtime.types import OptionContract, RuntimeManifest, StateCluster
from hitl_pmp.agentic_runtime.vision import VLMJudge


class NumericFixtures:
    @staticmethod
    def observation() -> dict[str, Any]:
        return {
            "observation_mode": "object_state",
            "objects": [
                {"name": "cube0", "type": "cube", "features": {"x": 0.4, "y": -0.2, "z": 0.03}},
                {"name": "bin0", "type": "bin", "features": {"x": 1.8, "width": 0.3}},
            ],
            "proprioception": {"pos_arm_joint1": 0.2, "gripper_command": 0.0},
            "action_spec": {"dimension": 18, "base_frame": "world"},
            "state_spec": {"position_frame": "world", "length_unit": "m"},
            "control_step": 12,
            "simulation_time_s": 1.2,
        }

    @staticmethod
    def option() -> OptionContract:
        return OptionContract(
            option_id="pick",
            belief_skill="pick",
            initiation="The cube is within reach and the gripper is empty.",
            termination="The robot stops or the controller reaches its step limit.",
            success="The gripper has lifted the cube off the floor.",
            controller="pick.py",
            cost=1,
        )

    @staticmethod
    def manifest() -> RuntimeManifest:
        return RuntimeManifest(
            revision="initial",
            clusters=(StateCluster(cluster_id="ready", description="The cube is within reach."),),
            options=(NumericFixtures.option(),),
            edges=(),
        )


class RecordingClient:
    def __init__(self, *, result: str) -> None:
        self.result = result
        self.messages: list[dict[str, Any]] = []

    def complete(self, *, messages: list[dict[str, Any]]) -> str:
        self.messages = messages
        return self.result


def test_measurements_and_generated_description_reach_classifier() -> None:
    client = RecordingClient(result='{"cluster_id":null,"reason":"Reachability is ambiguous."}')
    judge = VLMJudge(client=client, classification_prompt="classify", option_check_prompt="check")
    observation = NumericFixtures.observation()
    result = judge.classify(observation=observation, manifest=NumericFixtures.manifest())
    assert result.cluster_id is None
    assert len(client.messages[1]["content"]) == 1
    payload = json.loads(client.messages[1]["content"][0]["text"])
    assert payload["observation"] == observation
    assert payload["clusters"][0]["description"] == "The cube is within reach."


@pytest.mark.parametrize("phase", ["initiation", "termination", "success"])
def test_control_completion_never_substitutes_for_contract_judgment(*, phase: Any) -> None:
    client = RecordingClient(result='{"value":null,"reason":"Cannot establish the condition."}')
    judge = VLMJudge(client=client, classification_prompt="classify", option_check_prompt="check")
    observation = NumericFixtures.observation()
    observation["execution"] = {
        "steps": 12,
        "controller_done": True,
        "error": None,
        "trajectory_path": "rollout.jsonl",
        "seed": 7,
    }
    option = NumericFixtures.option()
    result = judge.check(observation=observation, option=option, phase=phase)
    assert result.value is None
    payload = json.loads(client.messages[1]["content"][0]["text"])
    assert payload["phase"] == phase
    assert payload["description"] == getattr(option, phase)
    assert payload["observation"]["execution"]["controller_done"] is True


@pytest.mark.parametrize("mode", [None, "rgb", "object", "object_states"])
def test_numeric_fields_do_not_implicitly_disable_image_requirement(*, mode: str | None) -> None:
    client = RecordingClient(result='{"cluster_id":"ready","reason":"should not be called"}')
    judge = VLMJudge(client=client, classification_prompt="classify", option_check_prompt="check")
    observation = NumericFixtures.observation()
    if mode is None:
        del observation["observation_mode"]
    else:
        observation["observation_mode"] = mode
    with pytest.raises(ValueError, match="actual image"):
        judge.classify(observation=observation, manifest=NumericFixtures.manifest())
    assert not client.messages


@pytest.mark.parametrize(
    "objects",
    [
        [],
        None,
        {},
        [{"name": "cube0", "type": "cube", "features": {}}],
        [{"name": "", "type": "cube", "features": {"x": 0.0}}],
        [{"name": "cube0", "type": " ", "features": {"x": 0.0}}],
        [{"name": "cube0", "type": "cube", "features": {"x": 0.0}, "success": True}],
        [{"name": "cube0", "type": "cube", "features": {"x": 0.0}}] * 2,
    ],
)
def test_missing_or_ambiguous_semantic_objects_are_rejected(*, objects: Any) -> None:
    client = RecordingClient(result='{"value":true,"reason":"should not be called"}')
    judge = VLMJudge(client=client, classification_prompt="classify", option_check_prompt="check")
    observation = NumericFixtures.observation()
    observation["objects"] = objects
    with pytest.raises(ValueError):
        judge.check(observation=observation, option=NumericFixtures.option(), phase="success")
    assert not client.messages


@pytest.mark.parametrize(
    "measurement", [float("nan"), float("inf"), -float("inf"), True, "0.3", None]
)
def test_invalid_measurements_are_rejected_before_inference(*, measurement: Any) -> None:
    client = RecordingClient(result='{"value":true,"reason":"should not be called"}')
    judge = VLMJudge(client=client, classification_prompt="classify", option_check_prompt="check")
    observation = NumericFixtures.observation()
    observation["objects"][0]["features"]["x"] = measurement
    with pytest.raises(ValueError, match="finite numbers"):
        judge.check(observation=observation, option=NumericFixtures.option(), phase="success")
    assert not client.messages


@pytest.mark.parametrize("location", ["top", "feature", "execution"])
def test_oracle_success_flags_are_rejected(*, location: str) -> None:
    client = RecordingClient(result='{"value":true,"reason":"should not be called"}')
    judge = VLMJudge(client=client, classification_prompt="classify", option_check_prompt="check")
    observation = NumericFixtures.observation()
    if location == "top":
        observation["success"] = True
    elif location == "feature":
        observation["objects"][0]["features"]["success"] = 1.0
    else:
        observation["execution"] = {"success": True}
    with pytest.raises(ValueError):
        judge.check(observation=observation, option=NumericFixtures.option(), phase="success")
    assert not client.messages


def test_object_condition_cannot_silently_include_images() -> None:
    client = RecordingClient(result='{"value":true,"reason":"should not be called"}')
    judge = VLMJudge(client=client, classification_prompt="classify", option_check_prompt="check")
    observation = NumericFixtures.observation()
    observation["images"] = ["data:image/png;base64,YQ=="]
    with pytest.raises(ValueError, match="undeclared fields"):
        judge.check(observation=observation, option=NumericFixtures.option(), phase="success")


def test_numeric_condition_keeps_strict_boolean_schema() -> None:
    client = RecordingClient(result='{"value":"true","reason":"Uncertain."}')
    judge = VLMJudge(client=client, classification_prompt="classify", option_check_prompt="check")
    with pytest.raises(ValidationError):
        judge.check(
            observation=NumericFixtures.observation(),
            option=NumericFixtures.option(),
            phase="success",
        )


def test_recent_physical_samples_reach_language_judgment_without_inferred_outcome() -> None:
    client = RecordingClient(result='{"value":null,"reason":"Motion remains ambiguous."}')
    judge = VLMJudge(client=client, classification_prompt="classify", option_check_prompt="check")
    before = NumericFixtures.observation()
    observation = NumericFixtures.observation()
    observation["objects"][0]["features"]["z"] = 0.3
    observation["recent_observations"] = [before]
    result = judge.check(observation=observation, option=NumericFixtures.option(), phase="success")
    assert result.value is None
    payload = json.loads(client.messages[1]["content"][0]["text"])
    assert payload["observation"]["recent_observations"] == [before]


@pytest.mark.parametrize("history", [[], [NumericFixtures.observation()] * 9, {}, [None]])
def test_history_requires_one_to_eight_object_observations(*, history: Any) -> None:
    client = RecordingClient(result='{"value":true,"reason":"should not be called"}')
    judge = VLMJudge(client=client, classification_prompt="classify", option_check_prompt="check")
    observation = NumericFixtures.observation()
    observation["recent_observations"] = history
    with pytest.raises(ValueError):
        judge.check(observation=observation, option=NumericFixtures.option(), phase="success")
    assert not client.messages


@pytest.mark.parametrize(
    "field,value",
    [
        ("recent_observations", [NumericFixtures.observation()]),
        ("execution", {"controller_done": True}),
        ("execution_seed", 7),
        ("success", True),
        ("observation_mode", "rgb"),
    ],
)
def test_history_rejects_nested_control_or_oracle_evidence(*, field: str, value: Any) -> None:
    client = RecordingClient(result='{"value":true,"reason":"should not be called"}')
    judge = VLMJudge(client=client, classification_prompt="classify", option_check_prompt="check")
    sample = NumericFixtures.observation()
    sample[field] = value
    observation = NumericFixtures.observation()
    observation["recent_observations"] = [sample]
    with pytest.raises(ValueError):
        judge.check(observation=observation, option=NumericFixtures.option(), phase="success")
    assert not client.messages


def test_history_validates_measurements_as_strictly_as_current_state() -> None:
    client = RecordingClient(result='{"value":true,"reason":"should not be called"}')
    judge = VLMJudge(client=client, classification_prompt="classify", option_check_prompt="check")
    sample = NumericFixtures.observation()
    sample["objects"][0]["features"]["z"] = float("nan")
    observation = NumericFixtures.observation()
    observation["recent_observations"] = [sample]
    with pytest.raises(ValueError, match="finite numbers"):
        judge.check(observation=observation, option=NumericFixtures.option(), phase="success")
    assert not client.messages


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), -0.1, True, "1.2"])
@pytest.mark.parametrize("location", ["current", "history"])
def test_timestamps_are_finite_nonnegative_measurements(*, timestamp: Any, location: str) -> None:
    client = RecordingClient(result='{"value":true,"reason":"should not be called"}')
    judge = VLMJudge(client=client, classification_prompt="classify", option_check_prompt="check")
    observation = NumericFixtures.observation()
    if location == "current":
        observation["simulation_time_s"] = timestamp
    else:
        sample = NumericFixtures.observation()
        sample["simulation_time_s"] = timestamp
        observation["recent_observations"] = [sample]
    with pytest.raises(ValueError, match="simulation_time_s"):
        judge.check(observation=observation, option=NumericFixtures.option(), phase="success")
    assert not client.messages
