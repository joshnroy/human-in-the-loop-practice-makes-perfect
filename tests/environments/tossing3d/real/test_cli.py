import json
from pathlib import Path
from typing import Any

import pytest

from hitl_pmp.cli import ENVIRONMENTS, STEP_PROTOCOL_ENVIRONMENTS, Cli
from hitl_pmp.environments.tossing3d.real.cli import Tossing3DRealCli
from hitl_pmp.environments.tossing3d.real.environment import RealTossing3DEnvironment
from hitl_pmp.environments.tossing3d.real.step_protocol import (
    InProcessEvaluations,
    Tossing3DRealStepProtocol,
)
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import REAL_TOSS_PARAMETERS
from hitl_pmp.step_protocol import StepPracticeRunner

FAKE_RUN = [
    *("--env", "tossing3d_real", "--method", "pomdp", "--preset", "tossing3d-canonical"),
    *("--robot", "fake", "--num-test-tasks", "1", "--seed", "0"),
]


def test_the_real_domain_is_registered_for_the_step_protocol() -> None:
    assert ENVIRONMENTS["tossing3d_real"] is Tossing3DRealCli
    assert STEP_PROTOCOL_ENVIRONMENTS["tossing3d_real"] is Tossing3DRealStepProtocol


def test_the_canonical_preset_applies_to_the_real_domain_with_weight_zero() -> None:
    args = Cli.parse_args(
        argv=[*FAKE_RUN, "--practice-step-budget", "100", "--output-dir", "unused"]
    )
    assert args.pomdp_observation_probability_weight == 0.0
    assert Path(args.practice_cost_config).name == "tossing3d_tick_priced_costs.json"
    assert args.max_steps_per_interaction == 20
    assert args.robot == "fake"
    assert args.initial_measurement is False


def test_the_real_problems_carry_the_real_toss_set_and_evaluation_skips_hard_reset() -> None:
    args = Cli.parse_args(
        argv=[*FAKE_RUN, "--practice-step-budget", "100", "--output-dir", "unused"]
    )
    practice = Tossing3DRealCli.build_practice_problem(args=args).env
    evaluation = Tossing3DRealCli.build_evaluation_problem(args=args).env
    assert isinstance(practice, RealTossing3DEnvironment)
    assert isinstance(evaluation, RealTossing3DEnvironment)
    assert practice.toss_parameters == REAL_TOSS_PARAMETERS
    assert not practice.evaluation and evaluation.evaluation


def test_real_evaluations_run_in_this_process(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def capture(**kwargs: Any) -> None:
        seen.update(kwargs)

    monkeypatch.setattr(StepPracticeRunner, "run", staticmethod(capture))
    Cli.main(
        argv=[*FAKE_RUN, "--practice-step-budget", "100", "--output-dir", str(tmp_path / "run")]
    )
    assert isinstance(seen["evaluations"], InProcessEvaluations)


def test_a_fake_robot_run_practises_and_ends_with_one_evaluation(*, tmp_path: Path) -> None:
    output = tmp_path / "run"
    Cli.main(argv=[*FAKE_RUN, "--practice-step-budget", "1500", "--output-dir", str(output)])
    events = [json.loads(line) for line in (output / "step_events.jsonl").read_text().splitlines()]
    measurements = [event for event in events if event["event"] == "measurement"]
    assert len(measurements) == 1
    assert measurements[0]["practice_steps"] > 0
    assert (output / "evaluations" / "0000").exists()
