"""A long practice run of a gated domain starts only on a commit whose skill benchmark
passed, unless the launch says, explicitly, to skip the gate."""

import json
from pathlib import Path

import pytest

from hitl_pmp.launch_gate import LaunchBlocked, LaunchGate, LaunchIdentity

COMMIT = "a" * 40
KG = "b" * 40
KM = "c" * 40
HERE = LaunchIdentity(
    git_commit=COMMIT, git_dirty=False, kindergarden_commit=KG, kinder_models_commit=KM
)


def _report(*, root: Path, commit: str = COMMIT, **overrides: object) -> Path:
    report = {
        "domain": "sweepoffshelf3d",
        "git_commit": commit,
        "git_dirty": False,
        "kindergarden_commit": KG,
        "kinder_models_commit": KM,
        "passed": True,
        "skills": [
            {"skill": "PickWiper", "passed": True, "successes": 3, "total": 3},
            {"skill": "SweepShelf", "passed": True, "successes": 9, "total": 30},
        ],
        **overrides,
    }
    path = LaunchGate.report_path(root=root, domain="sweepoffshelf3d", commit=commit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report))
    return path


def _check(*, root: Path, identity: LaunchIdentity = HERE, override: bool = False) -> str:
    return LaunchGate.check(
        domain="sweepoffshelf3d", root=root, identity=identity, override=override
    )


def test_a_passing_report_for_this_commit_lets_the_run_start(*, tmp_path: Path) -> None:
    path = _report(root=tmp_path)
    assert str(path) in _check(root=tmp_path)


def test_no_report_for_this_commit_blocks_the_run(*, tmp_path: Path) -> None:
    _report(root=tmp_path, commit="d" * 40)
    with pytest.raises(LaunchBlocked, match="no skill benchmark report"):
        _check(root=tmp_path)


def test_a_failing_report_blocks_and_names_the_skills_below_target(*, tmp_path: Path) -> None:
    _report(
        root=tmp_path,
        passed=False,
        skills=[
            {"skill": "PickWiper", "passed": False, "successes": 15, "total": 31},
            {"skill": "PlaceCubeInBox", "passed": True, "successes": 5, "total": 5},
        ],
    )
    with pytest.raises(LaunchBlocked, match=r"PickWiper gated 15/31") as blocked:
        _check(root=tmp_path)
    assert "PlaceCubeInBox" not in str(blocked.value)


def test_the_refusal_shows_each_failing_skills_gated_and_reported_only_counts(
    *, tmp_path: Path
) -> None:
    _report(
        root=tmp_path,
        passed=False,
        skills=[
            {
                "skill": "ReturnCubeToShelfFromBox",
                "target": "at least 3/8 of harvested",
                "passed": False,
                "successes": 7,
                "total": 21,
                "gated_successes": 2,
                "gated_total": 8,
                "reported_successes": 5,
                "reported_total": 13,
            },
            {
                "skill": "PickCubeFromShelf",
                "target": "at least 97/100",
                "passed": False,
                "successes": 99,
                "total": 99,
                "gated_successes": 99,
                "gated_total": 99,
                "reported_successes": 0,
                "reported_total": 0,
                "reason": "the benchmark ran 99 gated case(s), the threshold is out of 100",
            },
            {
                "skill": "PickCubeFromFloor",
                "target": "at least 11/12",
                "passed": True,
                "successes": 11,
                "total": 12,
                "gated_successes": 11,
                "gated_total": 12,
                "reported_successes": 0,
                "reported_total": 0,
            },
        ],
    )
    with pytest.raises(LaunchBlocked) as blocked:
        _check(root=tmp_path)
    message = str(blocked.value)
    assert (
        "ReturnCubeToShelfFromBox gated 2/8 (target at least 3/8 of harvested), "
        "reported-only 5/13" in message
    )
    assert "PickCubeFromShelf gated 99/99 (target at least 97/100)" in message
    assert "the benchmark ran 99 gated case(s), the threshold is out of 100" in message
    assert "PickCubeFromFloor" not in message
    assert "--skip-skill-benchmark-gate" in message


@pytest.mark.parametrize(
    ("identity", "match"),
    [
        (HERE.model_copy(update={"git_dirty": True}), "uncommitted changes"),
        (HERE.model_copy(update={"kindergarden_commit": "e" * 40}), "kindergarden"),
        (HERE.model_copy(update={"kinder_models_commit": "e" * 40}), "kinder-models"),
        (HERE.model_copy(update={"git_commit": "unknown"}), "cannot tell which commit"),
    ],
)
def test_a_report_that_does_not_describe_this_code_blocks(
    *, tmp_path: Path, identity: LaunchIdentity, match: str
) -> None:
    _report(root=tmp_path, commit=COMMIT)
    with pytest.raises(LaunchBlocked, match=match):
        _check(root=tmp_path, identity=identity)


def test_a_report_benchmarked_on_a_dirty_tree_does_not_count(*, tmp_path: Path) -> None:
    _report(root=tmp_path, git_dirty=True)
    with pytest.raises(LaunchBlocked, match="dirty"):
        _check(root=tmp_path)


def test_the_explicit_override_starts_the_run_and_says_so(*, tmp_path: Path) -> None:
    message = _check(root=tmp_path, override=True)
    assert "SKIPPED" in message


def test_an_ungated_domain_is_never_blocked(*, tmp_path: Path) -> None:
    assert (
        LaunchGate.check(domain="lightswitch", root=tmp_path, identity=HERE, override=False) == ""
    )


def _gate_here(*, monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(LaunchGate, "default_root", staticmethod(lambda: root))
    monkeypatch.setattr(LaunchIdentity, "current", staticmethod(lambda: HERE))


SHELF_STEP_RUN = [
    "--env",
    "sweepoffshelf3d",
    "--method",
    "ees",
    "--practice-step-budget",
    "5",
]


def test_the_cli_refuses_a_shelf_practice_run_before_building_anything(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hitl_pmp.cli import Cli
    from hitl_pmp.step_protocol import StepProtocolEntry

    _gate_here(monkeypatch=monkeypatch, root=tmp_path / "reports")
    started: list[object] = []
    monkeypatch.setattr(StepProtocolEntry, "run", staticmethod(lambda **kw: started.append(kw)))
    with pytest.raises(LaunchBlocked, match="no skill benchmark report"):
        Cli.main(argv=[*SHELF_STEP_RUN, "--output-dir", str(tmp_path / "run")])
    assert started == []
    assert not (tmp_path / "run").exists()
    Cli.main(
        argv=[*SHELF_STEP_RUN, "--output-dir", str(tmp_path / "run"), "--skip-skill-benchmark-gate"]
    )
    assert len(started) == 1
    _report(root=tmp_path / "reports")
    Cli.main(argv=[*SHELF_STEP_RUN, "--output-dir", str(tmp_path / "run")])
    assert len(started) == 2


def test_run_sweep_refuses_a_shelf_sweep_before_spawning_and_forwards_the_override(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from scripts import run_sweep

    _gate_here(monkeypatch=monkeypatch, root=tmp_path / "reports")
    executed: list[list[object]] = []
    monkeypatch.setattr(
        run_sweep.SweepRunner,
        "execute",
        staticmethod(lambda **kw: executed.append(kw["runs"]) or []),
    )
    argv = [
        "run_sweep.py",
        "--env",
        "sweepoffshelf3d",
        "--methods",
        "ees",
        "--num-seeds",
        "1",
        "--results-root",
        str(tmp_path / "results"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(LaunchBlocked):
        run_sweep.main()
    assert executed == []
    monkeypatch.setattr(sys, "argv", [*argv, "--skip-skill-benchmark-gate"])
    run_sweep.main()
    (runs,) = executed
    assert all("--skip-skill-benchmark-gate" in run.command for run in runs)
