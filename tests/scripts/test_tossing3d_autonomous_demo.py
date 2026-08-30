"""The demo delegates all decisions to EES and refuses reset-assisted evidence."""

from pathlib import Path

import pytest

from hitl_pmp.cli import Cli
from scripts.tossing3d_autonomous_demo import AutonomousDemo


def test_recipe_uses_ees_and_keeps_practice_continuous(*, tmp_path: Path) -> None:
    args = Cli.parse_args(
        argv=AutonomousDemo.arguments(output_dir=tmp_path, cycles=2, steps=8, seed=0)
    )
    assert args.method == "ees"
    assert args.layout == "same-side"
    assert args.practice_reset_policy == "never"
    assert args.practice_reset_interval is None
    assert args.ask_for_reset_cube_bin_cost is None
    assert args.num_cycles == 2
    assert args.max_steps_per_interaction == 8
    assert args.record_sampler_draws
    assert args.record_skill_competence


def test_recipe_delegates_to_the_standard_cli(
    *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []
    monkeypatch.setattr(Cli, "main", lambda *, argv: calls.append(argv))
    AutonomousDemo.run(output_dir=tmp_path, cycles=1, steps=16, seed=0)
    assert calls == [AutonomousDemo.arguments(output_dir=tmp_path, cycles=1, steps=16, seed=0)]


@pytest.mark.parametrize("cycles,steps", [(0, 16), (1, 0), (-1, 10)])
def test_reject_empty_sessions(*, tmp_path: Path, cycles: int, steps: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        AutonomousDemo.arguments(output_dir=tmp_path, cycles=cycles, steps=steps, seed=0)


def test_evidence_excludes_unscored_actions_and_refuses_resets() -> None:
    from analysis.tossing3d_autonomous import AutonomousEvidence

    stats = {
        "num_practice_resets": 0,
        "num_human_interventions_recorded": 0,
        "evaluations": [[0, 0, 1], [4, 0, 1]],
        "practice_outcomes_per_cycle": [
            {
                "PickCubeFromFloor": {"num_attempts": 2, "num_successes": 1},
                "MoveToTossLocationAndToss": {"num_attempts": 1, "num_successes": 0},
            }
        ],
    }
    summary = AutonomousEvidence.summarize(stats=stats)
    assert summary["executed_actions"] == 4
    assert summary["scored_actions"] == 3
    assert summary["unscored_actions"] == 1
    assert summary["skills"]["PickCubeFromFloor"] == {"attempts": 2, "successes": 1}
    stats["num_practice_resets"] = 1
    with pytest.raises(ValueError, match="reset"):
        AutonomousEvidence.summarize(stats=stats)
    stats["num_practice_resets"] = 0
    stats["num_human_interventions_recorded"] = 1
    with pytest.raises(ValueError, match="human"):
        AutonomousEvidence.summarize(stats=stats)
