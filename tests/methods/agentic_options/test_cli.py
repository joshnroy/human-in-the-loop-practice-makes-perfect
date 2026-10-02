"""Experiment configuration is trusted input, never inferred from generated code."""

import copy
from pathlib import Path

import pytest

from hitl_pmp.agentic_runtime.types import GeneratedLibrary
from hitl_pmp.methods.agentic_options.cli import AgenticOptionsCli, RuntimeConfig
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    RESET_SKILL,
    TOSS_SKILL,
)

from .support import library


def test_vision_inherits_existing_broker_but_uses_current_run_paths(*, tmp_path: Path) -> None:
    config = RuntimeConfig.model_validate({
        "coding": {
            "sandbox": {
                "image": "strict:test",
                "robocode_checkout": str(tmp_path / "robocode"),
                "artifact_dir": str(tmp_path / "policies"),
            },
            "artifact_dir": str(tmp_path / "coding"),
            "backend": "codex",
            "model": "configured-multimodal-model",
            "max_budget_usd": 2,
            "reasoning_effort": "high",
        },
        "vision": {
            "provider": "robocode_broker",
            "artifact_dir": "stale-run",
            "robocode_checkout": "wrong-checkout",
        },
    })
    resolved = config.resolved_vision(artifact_dir=tmp_path / "current-run")
    assert resolved["backend"] == config.coding.backend
    assert resolved["model"] == config.coding.model
    assert resolved["artifact_dir"] == str(tmp_path / "current-run")
    assert resolved["robocode_checkout"] == str(tmp_path / "robocode")
    assert "model" not in config.vision
    config.vision["model"] = "explicit-vision-model"
    assert config.resolved_vision(artifact_dir=tmp_path)["model"] == "explicit-vision-model"


def complete_library() -> GeneratedLibrary:
    manifest = library().model_dump(mode="json")
    for name, skill in (("pick", PICK_SKILL), ("open_gripper", OPEN_GRIPPER_SKILL)):
        manifest["options"].append({
            "option_id": name,
            "belief_skill": skill,
            "initiation": "Fixture ready.",
            "termination": "Fixture ended.",
            "success": "Fixture result.",
            "controller": f"policies/{name}.py",
            "cost": 123.0,
        })
    return GeneratedLibrary.model_validate({
        "manifest": manifest,
        "controllers": {
            option["controller"]: "# Not executed; hand-written configuration fixture.\n"
            for option in manifest["options"]
            if option.get("controller")
        },
    })


def input_bundle() -> dict:
    return {
        "configured_costs": {"robot": 1.0, "human": 5.0},
        "belief_skills": {
            "pick": PICK_SKILL,
            "toss": TOSS_SKILL,
            "open_gripper": OPEN_GRIPPER_SKILL,
            "human_reset": RESET_SKILL,
        },
        "robot_options": {
            "pick": PICK_SKILL,
            "toss": TOSS_SKILL,
            "open_gripper": OPEN_GRIPPER_SKILL,
        },
        "human_interventions": {"reset_near": "robot_side", "reset_far": "opposite_side"},
    }


def test_no_human_arm_removes_interventions_and_their_graph_edges() -> None:
    generated = complete_library()
    before = generated.model_dump_json()
    configured = AgenticOptionsCli.configure_library(
        library=generated, bundle=input_bundle(), human_enabled=False, human_cost=9.0
    )

    assert {option.option_id for option in configured.manifest.options} == {
        "pick",
        "toss",
        "open_gripper",
    }
    assert all(not edge.option_id.startswith("reset") for edge in configured.manifest.edges)
    assert configured.controllers == generated.controllers
    assert generated.model_dump_json() == before


def test_generated_costs_cannot_override_configured_accounting() -> None:
    configured = AgenticOptionsCli.configure_library(
        library=complete_library(), bundle=input_bundle(), human_enabled=True, human_cost=7.5
    )
    assert {item.cost for item in configured.manifest.options if item.controller} == {1.0}
    assert {item.cost for item in configured.manifest.options if item.human_destination} == {7.5}


@pytest.mark.parametrize("which", ["robot_skill", "human_skill", "human_mechanism"])
def test_generated_library_cannot_invent_belief_or_human_mechanisms(*, which: str) -> None:
    value = complete_library().model_dump()
    option = value["manifest"]["options"][0 if which == "robot_skill" else 1]
    if which == "human_mechanism":
        old_id = option["option_id"]
        option["option_id"] = "free_magic_reset"
        for edge in value["manifest"]["edges"]:
            if edge["option_id"] == old_id:
                edge["option_id"] = option["option_id"]
    else:
        option["belief_skill"] = "LLMInventedCompetenceModel"
    generated = GeneratedLibrary.model_validate(value)

    with pytest.raises(ValueError):
        AgenticOptionsCli.configure_library(
            library=generated, bundle=input_bundle(), human_enabled=True, human_cost=5.0
        )


def test_canonical_deployment_rejects_a_missing_recovery_skill() -> None:
    value = copy.deepcopy(complete_library().model_dump())
    value["manifest"]["options"] = [
        item for item in value["manifest"]["options"] if item["option_id"] != "open_gripper"
    ]
    del value["controllers"]["policies/open_gripper.py"]
    with pytest.raises(ValueError, match="requires pick, toss and open-gripper"):
        AgenticOptionsCli.configure_library(
            library=GeneratedLibrary.model_validate(value),
            bundle=input_bundle(),
            human_enabled=True,
            human_cost=5.0,
        )


def test_generated_options_cannot_swap_existing_belief_slots() -> None:
    value = complete_library().model_dump()
    options = {item["option_id"]: item for item in value["manifest"]["options"]}
    options["pick"]["belief_skill"] = TOSS_SKILL
    options["toss"]["belief_skill"] = PICK_SKILL

    with pytest.raises(ValueError):
        AgenticOptionsCli.configure_library(
            library=GeneratedLibrary.model_validate(value),
            bundle=input_bundle(),
            human_enabled=True,
            human_cost=5.0,
        )


def test_generated_option_cannot_borrow_a_known_belief_under_an_unlisted_id() -> None:
    value = complete_library().model_dump(mode="json")
    options = value["manifest"]["options"]
    extra = copy.deepcopy(next(item for item in options if item["option_id"] == "pick"))
    extra["option_id"] = "unlisted_pick"
    extra["controller"] = "policies/unlisted_pick.py"
    options.append(extra)
    value["controllers"][extra["controller"]] = "# Hand-written unit fixture, never executed.\n"

    with pytest.raises(ValueError):
        AgenticOptionsCli.configure_library(
            library=GeneratedLibrary.model_validate(value),
            bundle=input_bundle(),
            human_enabled=True,
            human_cost=5.0,
        )
