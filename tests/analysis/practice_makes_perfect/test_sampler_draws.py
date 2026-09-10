"""Covers the reader for `--record-sampler-draws`' per-draw file.

Built from hand-written fixtures rather than by running the CLI: the writer's own
end-to-end behaviour is `tests/test_sampler_draws.py`'s job, and a reader test that
had to run EES first would be slow and would fail for the writer's reasons. The two
files share `SamplerDraw`, so the schema cannot drift between them.
"""

import json
from pathlib import Path

from analysis.practice_makes_perfect.sampler_draws import SamplerDrawAnalysis
from hitl_pmp.sampler_draws import SAMPLER_DRAWS_FILENAME, SamplerDraw


def _draw(
    *, cycle: int, skill: str = "MoveToThrowPose", value: float, success: bool, pool: str
) -> str:
    return SamplerDraw(
        cycle=cycle,
        skill=skill,
        consultation=pool,
        success=success,
        params=[value],
        achieved={"robot.pos_base_x": 1.0 + value, "bin.x": 2.5},
    ).model_dump_json()


def _write_run(*, run_dir: Path, lines: list[str], evaluations: list[list[int]]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / SAMPLER_DRAWS_FILENAME).write_text("\n".join(lines) + "\n")
    (run_dir / "stats.json").write_text(json.dumps({"evaluations": evaluations}))
    return run_dir


def test_load_run_reads_every_draw_in_order(*, tmp_path: Path) -> None:
    run = _write_run(
        run_dir=tmp_path / "ees" / "0",
        lines=[
            _draw(cycle=0, value=0.5, success=False, pool="uninformative"),
            _draw(cycle=1, value=1.3, success=True, pool="informed"),
        ],
        evaluations=[[0, 0, 10], [4, 3, 10]],
    )
    draws = SamplerDrawAnalysis.load_run(run_dir=run)
    assert [d.cycle for d in draws] == [0, 1]
    assert [d.params[0] for d in draws] == [0.5, 1.3]


def test_a_truncated_trailing_line_does_not_break_the_read(*, tmp_path: Path) -> None:
    """The reason the format is JSONL: a 100-cycle run killed mid-flight must still
    be readable up to its last flushed draw. A reader that choked on the blank tail
    would give that robustness away."""
    run = tmp_path / "ees" / "0"
    run.mkdir(parents=True)
    # Whitespace, not just an empty string. A kill between `write` and `flush` can
    # leave a partial line, and `splitlines()` on a file ending "\n   " yields "   ",
    # which a bare `if line` filter would hand to the JSON parser.
    (run / SAMPLER_DRAWS_FILENAME).write_text(
        _draw(cycle=0, value=0.5, success=True, pool="informed") + "\n   \n"
    )
    assert len(SamplerDrawAnalysis.load_run(run_dir=run)) == 1


def test_pool_counts_are_successes_over_attempts(*, tmp_path: Path) -> None:
    """The x/y pairs every report prints. Kept as a pair rather than a rate so a
    denominator cannot go missing between here and the page."""
    run = _write_run(
        run_dir=tmp_path / "ees" / "0",
        lines=[
            _draw(cycle=0, value=0.5, success=False, pool="uninformative"),
            _draw(cycle=0, value=0.6, success=True, pool="uninformative"),
            _draw(cycle=1, value=1.3, success=True, pool="informed"),
        ],
        evaluations=[[0, 0, 10], [4, 3, 10]],
    )
    counts = SamplerDrawAnalysis.pool_counts(draws=SamplerDrawAnalysis.load_run(run_dir=run))
    assert counts["MoveToThrowPose"]["uninformative"] == (1, 2)
    assert counts["MoveToThrowPose"]["informed"] == (1, 1)


def test_transitions_by_cycle_joins_stats_json(*, tmp_path: Path) -> None:
    """A draw carries a cycle, and every learning curve is drawn against online
    transitions; this is the only place the two are joined."""
    run = _write_run(
        run_dir=tmp_path / "ees" / "0",
        lines=[_draw(cycle=0, value=0.5, success=True, pool="informed")],
        evaluations=[[0, 0, 10], [17, 3, 10], [31, 6, 10]],
    )
    assert SamplerDrawAnalysis.transitions_by_cycle(run_dir=run) == [0, 17, 31]


def test_a_run_whose_draws_file_holds_nothing_is_omitted_rather_than_reported_empty(
    *, tmp_path: Path
) -> None:
    """Two ways a run contributes no draws, and both must be omitted rather than
    keyed to an empty list -- an arm reported as `0/0` invites a comparison against
    an arm that measured something, which is a different statement entirely.

    The second case is the one that needs the guard: a run killed before its first
    flush completed leaves a file that exists and yields nothing. The first is
    handled by the glob, since `random-skills` consults no sampler and the recorder
    opens lazily, so no file is ever created.
    """
    _write_run(
        run_dir=tmp_path / "ees" / "0",
        lines=[_draw(cycle=0, value=0.5, success=True, pool="informed")],
        evaluations=[[0, 0, 10]],
    )
    no_file = tmp_path / "random-skills" / "0"
    no_file.mkdir(parents=True)
    (no_file / "stats.json").write_text(json.dumps({"evaluations": [[0, 0, 10]]}))
    empty_file = tmp_path / "ees" / "1"
    empty_file.mkdir(parents=True)
    (empty_file / SAMPLER_DRAWS_FILENAME).write_text("\n")
    (empty_file / "stats.json").write_text(json.dumps({"evaluations": [[0, 0, 10]]}))

    loaded = SamplerDrawAnalysis.load(results_root=tmp_path)
    assert set(loaded) == {("ees", 0)}


def test_parameter_trajectory_carries_cycle_value_outcome_and_pool(*, tmp_path: Path) -> None:
    """The four columns a trajectory figure needs: when, what was chosen, whether it
    worked, and whether the classifier had anything to say when it chose."""
    run = _write_run(
        run_dir=tmp_path / "ees" / "0",
        lines=[
            _draw(cycle=0, value=0.5, success=False, pool="uninformative"),
            _draw(cycle=2, skill="Pick", value=0.55, success=True, pool="informed"),
            _draw(cycle=3, value=1.34, success=True, pool="informed"),
        ],
        evaluations=[[0, 0, 10]],
    )
    draws = SamplerDrawAnalysis.load_run(run_dir=run)
    trajectory = SamplerDrawAnalysis.parameter_trajectory(draws=draws, skill="MoveToThrowPose")
    assert trajectory == [(0, 0.5, False, "uninformative"), (3, 1.34, True, "informed")]


def test_achieved_reads_a_post_action_feature_and_skips_draws_without_it(*, tmp_path: Path) -> None:
    """A missing key means the skill never bound that object, which is not a zero."""
    run = tmp_path / "ees" / "0"
    run.mkdir(parents=True)
    with_feature = _draw(cycle=0, value=0.5, success=True, pool="informed")
    without = SamplerDraw(
        cycle=1,
        skill="MoveToThrowPose",
        consultation="informed",
        success=True,
        params=[0.7],
        achieved={"bin.x": 2.5},
    ).model_dump_json()
    (run / SAMPLER_DRAWS_FILENAME).write_text(with_feature + "\n" + without + "\n")

    draws = SamplerDrawAnalysis.load_run(run_dir=run)
    achieved = SamplerDrawAnalysis.achieved(
        draws=draws, skill="MoveToThrowPose", feature="robot.pos_base_x"
    )
    assert achieved == [(0, 1.5)]
