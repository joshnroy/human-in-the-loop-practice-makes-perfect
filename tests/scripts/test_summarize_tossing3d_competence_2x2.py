"""Completion checks and compact exports for the streamed 2x2 run summaries."""

import json
from pathlib import Path

import pytest

from scripts.summarize_tossing3d_competence_2x2 import (
    ARMS,
    TOSS_SKILL,
    iter_jsonl,
    plot_summary,
    summarize_results,
    write_summary,
)


def _write_jsonl(*, path: Path, rows: list[dict]) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def _completed_tree(*, root: Path) -> None:
    commands = {f"{arm}/seed_00": [] for arm in ARMS}
    (root / "manifest.json").write_text(
        json.dumps({
            "commands": commands,
            "num_cycles": 2,
            "num_seeds": 1,
            "source_commit": "test-source",
        })
    )
    (root / "run_status.json").write_text(
        json.dumps([{"name": name, "returncode": 0} for name in commands])
    )
    for name in commands:
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "stats.json").write_text(
            json.dumps({
                "evaluations": [[0, 0, 10], [1, 1, 10], [2, 2, 10]],
                "practice_session_ends": [
                    {"cycle_index": cycle, "reason": "planner_stop", "actions_executed": 1}
                    for cycle in range(2)
                ],
                "practice_outcomes_per_cycle": [
                    {TOSS_SKILL: {"num_attempts": 1, "num_unparameterized_attempts": 0}}
                    for _ in range(2)
                ],
                "num_practice_resets": 0,
                "num_human_interventions_recorded": 0,
                "summed_human_cost_recorded": 0,
            })
        )
        _write_jsonl(
            path=directory / "progress.jsonl",
            rows=[
                {
                    "sweeps_completed": cycle + 1,
                    "sweeps_total": 3,
                    "transitions": cycle,
                    "num_solved": cycle,
                    "num_total": 10,
                    "elapsed_seconds": 10 * cycle,
                }
                for cycle in range(3)
            ],
        )
        history = []
        decisions = []
        for cycle in range(2):
            competence = 0.2 + cycle / 10
            history.append({
                "cycle_index": cycle,
                "total_training_examples": cycle,
                "filtered_competence": competence,
                "smoothed_competence": competence,
                "filtered_learning_rate": 0.05,
                "smoothed_learning_rate": 0.05,
                "effective_sample_size": 800,
                "unique_ancestors": 900 if "particle" in name else None,
            })
            diagnostic = {
                TOSS_SKILL: {"effective_sample_size": 800, "unique_phi_configurations": 23}
            }
            decisions.extend([
                {
                    "event": "dispatch",
                    "cycle": cycle,
                    "skill": TOSS_SKILL,
                    "cost": 1.0,
                    "summed_cost": 1.0,
                },
                {
                    "event": "outcome",
                    "cycle": cycle,
                    "skill": TOSS_SKILL,
                    "success": cycle == 1,
                    "beliefs": diagnostic,
                    "belief": {"large_state": "x" * 100000},
                },
                {
                    "event": "smoothing",
                    "cycle": cycle,
                    "use": "retrospective_diagnostics_only",
                    "history": {TOSS_SKILL: list(history)},
                },
                {
                    "event": "refit",
                    "cycle": cycle,
                    "learning_rate_evidence": "success_failure_only",
                    "beliefs": diagnostic,
                    "training_examples": {TOSS_SKILL: 1},
                },
            ])
        _write_jsonl(path=directory / "pomdp_decisions.jsonl", rows=decisions)
        _write_jsonl(
            path=directory / "sampler_draws.jsonl",
            rows=[
                {
                    "cycle": cycle,
                    "skill": TOSS_SKILL,
                    "success": cycle == 1,
                    "consultation": "informed",
                }
                for cycle in range(2)
            ],
        )


def test_completed_summary_validates_counts_and_exports_compact_artifacts(
    *, tmp_path: Path
) -> None:
    _completed_tree(root=tmp_path)
    summary = summarize_results(results_root=tmp_path)
    assert summary["valid"]
    assert len(summary["runs"]) == 4
    for run in summary["runs"]:
        assert run["total_dispatched_cost"] == 2
        assert run["total_dispatches"] == 2
        assert run["refit_count"] == run["smoothing_count"] == run["practice_session_count"] == 2
        assert run["sampler_by_skill"][TOSS_SKILL] == {"attempts": 2, "successes": 1}
        assert run["ancestry_diagnostics"][TOSS_SKILL]["phi_configurations_final_filtered"] == 23
    output = tmp_path / "summary"
    write_summary(summary=summary, output_dir=output)
    plot_summary(summary=summary, output_dir=output)
    assert (output / "summary.json").stat().st_size < 50000
    assert len((output / "summary.csv").read_text().splitlines()) == 5
    assert (output / "comparison.png").read_bytes().startswith(b"\x89PNG")
    assert (output / "comparison.pdf").read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize(
    "damage", ["missing_refit", "failed_status", "free_reset", "missing_sampler"]
)
def test_incomplete_or_inconsistent_runs_are_rejected(*, tmp_path: Path, damage: str) -> None:
    _completed_tree(root=tmp_path)
    directory = tmp_path / ARMS[0] / "seed_00"
    if damage == "missing_refit":
        path = directory / "pomdp_decisions.jsonl"
        rows = list(iter_jsonl(path=path))
        _write_jsonl(path=path, rows=rows[:-1])
    elif damage == "failed_status":
        path = tmp_path / "run_status.json"
        statuses = json.loads(path.read_text())
        statuses[0]["returncode"] = 1
        path.write_text(json.dumps(statuses))
    elif damage == "free_reset":
        path = directory / "stats.json"
        stats = json.loads(path.read_text())
        stats["num_practice_resets"] = 1
        path.write_text(json.dumps(stats))
    else:
        (directory / "sampler_draws.jsonl").unlink()
    summary = summarize_results(results_root=tmp_path)
    assert not summary["valid"]
    assert summary["errors"]
    with pytest.raises(ValueError, match="incomplete"):
        plot_summary(summary=summary, output_dir=tmp_path)


def test_manifestless_smoke_requires_explicit_cycle_count(*, tmp_path: Path) -> None:
    _completed_tree(root=tmp_path)
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(ValueError, match="explicit"):
        summarize_results(results_root=tmp_path)
    assert summarize_results(results_root=tmp_path, num_cycles=2)["valid"]


def test_jsonl_reader_is_incremental_and_reports_truncated_tail(*, tmp_path: Path) -> None:
    path = tmp_path / "large.jsonl"
    path.write_text('{"event":"first"}\n{"incomplete":')
    records = iter_jsonl(path=path)
    assert next(records) == {"event": "first"}
    with pytest.raises(ValueError, match="large.jsonl:2"):
        next(records)
