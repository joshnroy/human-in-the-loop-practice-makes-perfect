"""`--env tossing3d_real --preset tossing3d-canonical`: the real arm runs the canonical
planner, and its tick costs come from `RealSkillStepCosts`, never a second table."""

import json
from pathlib import Path

import pytest

import hitl_pmp.environments.tossing3d.real.cli as real_cli
from hitl_pmp.cli import Cli
from hitl_pmp.core.practice_costs import PracticeCosts
from hitl_pmp.environments.tossing3d.real.cli import RealWorldRegistry
from hitl_pmp.environments.tossing3d.real.types import RealSkillStepCosts
from hitl_pmp.presets.preset import PRESET_DIRECTORY

pytestmark = pytest.mark.xdist_group("tossing3d_real_cli")

REAL_ARGV = ["--env", "tossing3d_real", "--method", "pomdp", "--robot-plant", "fake"]
PRESET = ["--preset", "tossing3d-canonical"]


@pytest.fixture(autouse=True)
def _fresh_registry():
    RealWorldRegistry.clear()
    yield
    RealWorldRegistry.clear()


def test_the_real_arm_through_the_preset_has_w_zero_and_the_tick_priced_costs() -> None:
    args = Cli.parse_args(argv=REAL_ARGV + PRESET + ["--practice-step-budget", "6000"])
    assert args.pomdp_observation_probability_weight == 0.0
    assert args.pomdp_linear_cost_lambda == 3e-6
    assert (
        args.practice_cost_config
        == (PRESET_DIRECTORY / "tossing3d_tick_priced_costs.json").resolve()
    )
    assert args.goal_pursuit_horizon == 0
    assert args.max_steps_per_interaction == 20
    # The robot's own defaults survive the preset: evaluation at the end only.
    assert args.measurement_interval_steps == real_cli.END_ONLY_MEASUREMENT_INTERVAL
    assert args.initial_measurement is False


def test_the_real_arm_without_the_preset_keeps_the_point_one_default() -> None:
    args = Cli.parse_args(argv=REAL_ARGV)
    assert args.pomdp_observation_probability_weight == 0.1
    assert args.practice_cost_config is None


def test_the_documented_real_launch_uses_the_preset_and_passes_w_zero_explicitly() -> None:
    launch = (real_cli.__doc__ or "").split("\n\n")[1]
    assert "--preset tossing3d-canonical" in launch
    assert "--pomdp-observation-probability-weight 0.0" in launch


def test_a_fake_plant_run_through_the_preset_practises_at_tick_prices(*, tmp_path: Path) -> None:
    Cli.main(
        argv=REAL_ARGV
        + PRESET
        + [
            "--seed",
            "0",
            "--practice-step-budget",
            "300",
            "--num-test-tasks",
            "1",
            "--evaluation-control-steps",
            "20",
            "--output-dir",
            str(tmp_path),
        ]
    )
    decisions = [
        json.loads(line) for line in (tmp_path / "pomdp_decisions.jsonl").read_text().splitlines()
    ]
    root = next(record for record in decisions if record["event"] == "decision")
    assert root["action"] != "STOP"
    assert root["observation_probability_weight"] == 0.0

    costs = PracticeCosts.load(path=tmp_path / "practice_costs.json")
    robot = [
        json.loads(line)
        for line in (tmp_path / "step_events.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "robot"
    ]
    pick = next(event for event in robot if event["label"].startswith("PickCube("))
    # One cost per tick, and a real pick's ticks are RealSkillStepCosts', so the preset
    # needs no copy of the 119/90/1 table to charge them.
    assert pick["end_step"] - pick["start_step"] == RealSkillStepCosts().pick_cube
    assert pick["cost"] == costs.robot_step.value * RealSkillStepCosts().pick_cube
    toss = next(e for e in robot if e["label"].startswith("MoveToTossLocationAndToss("))
    assert (
        toss["cost"] == costs.robot_step.value * RealSkillStepCosts().move_to_toss_location_and_toss
    )
