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
    far_tosses = [
        event
        for event in events
        if event["event"] == "robot"
        and str(event.get("label", "")).startswith("MoveToTossLocationAndToss")
        and "opposite_side)" in str(event["label"])
    ]
    assert far_tosses
    # The real standoff window keeps the base clear of the tape at the far spot, so no
    # far-side toss is refused (a refused toss is charged no steps).
    assert all(event["end_step"] > event["start_step"] for event in far_tosses)
    assert (output / "evaluations" / "0000").exists()
    session = (output / "robot_session.log").read_text().splitlines()
    assert any("cycle 0 [" in line and "action 1/20 · steps 0/1500" in line for line in session)
    assert any("running PickCube" in line for line in session)
    assert any("scene: FAR" in line for line in session)


def test_measuring_after_every_cycle_learns_then_measures_then_restores_practice(
    *, tmp_path: Path
) -> None:
    output = tmp_path / "run"
    Cli.main(
        argv=[
            *FAKE_RUN,
            *("--practice-step-budget", "1500", "--measure-every-cycles", "1"),
            # Short cycles, so a small budget holds several.
            *("--max-steps-per-interaction", "4"),
            *("--output-dir", str(output)),
        ]
    )
    events = [json.loads(line) for line in (output / "step_events.jsonl").read_text().splitlines()]
    kinds = [event["event"] for event in events]
    updates = [i for i, kind in enumerate(kinds) if kind == "learning_update"]
    assert len(updates) >= 2
    for position in updates:
        # Learn, then measure the policy just updated, then put practice's scene back.
        update, measurement, restored = events[position : position + 3]
        assert measurement["event"] == "measurement"
        assert measurement["after_cycle"] == update["cycle"]
        assert measurement["learning_updates"] == updates.index(position) + 1
        assert restored["event"] == "practice_scene_restored"
        assert restored["cycle"] == update["cycle"]
        assert restored["practice_steps"] == measurement["practice_steps"]
        # Practice carries on from the scene the person restored.
        assert restored["observed"]["cube"] == pytest.approx(restored["scene"]["cube"], abs=1e-6)
        assert restored["observed"]["bin_side"] == restored["scene"]["bin_side"]
        assert kinds[position + 3] in {"session_start", "practice_end", "measurement"}
    measurements = [event for event in events if event["event"] == "measurement"]
    cycle_measurements = [m for m in measurements if "after_cycle" in m]
    assert len(cycle_measurements) == len(updates)
    # No policy is measured twice: the final measurement runs only if an update came
    # after the last cycle's (a cycle the budget cut short, learned from), and no
    # restore follows it.
    finals = measurements[len(cycle_measurements) :]
    assert measurements[: len(cycle_measurements)] == cycle_measurements
    assert len(finals) <= 1 and all("after_cycle" not in m for m in finals)
    assert len({m["learning_updates"] for m in measurements}) == len(measurements)
    assert len({m["practice_steps"] for m in measurements}) == len(measurements)
    assert kinds.count("practice_scene_restored") == len(updates)
    assert kinds[-1] == "practice_end"
    evaluated = json.loads((output / "step_evaluations.json").read_text())
    assert len(evaluated) == len(measurements)
    assert all(record["complete"] for record in evaluated)
    session = (output / "robot_session.log").read_text().splitlines()
    assert any("eval 0 after cycle 0: task 1/1" in line for line in session)
    assert all(f"eval {len(updates)} (final): task 1/1" in "\n".join(session) for _ in finals)
    assert sum("restore practice:" in line for line in session) == len(updates)
    # A restore is the person's, not a human reset the method chose.
    assert kinds.count("human") == sum(" human reset: " in line for line in session)


def test_the_simulated_domain_has_no_measure_every_cycles_flag() -> None:
    with pytest.raises(SystemExit):
        Cli.parse_args(
            argv=[
                *("--env", "tossing3d", "--method", "pomdp", "--preset", "tossing3d-canonical"),
                *("--practice-step-budget", "100", "--output-dir", "unused"),
                *("--measure-every-cycles", "1"),
            ]
        )


def test_measuring_every_cycle_is_off_by_default_on_the_real_domain() -> None:
    args = Cli.parse_args(
        argv=[*FAKE_RUN, "--practice-step-budget", "100", "--output-dir", "unused"]
    )
    assert args.measure_every_cycles is None
