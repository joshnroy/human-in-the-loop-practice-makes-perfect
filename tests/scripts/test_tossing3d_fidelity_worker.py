import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import tossing3d_fidelity_worker as worker
from scripts.tossing3d_ees_fidelity_launch import build_manifest


@pytest.fixture
def manifest_file(*, tmp_path):
    manifest = build_manifest(
        size="pilot", goal_horizon=5, results_root=tmp_path / "results with ; $literal"
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


@pytest.fixture
def execution(*, monkeypatch, manifest_file):
    path, manifest = manifest_file
    source = {"head": manifest["source"]["head"], "tree": "b" * 40, "clean_tracked": True}
    check = Mock(return_value=source)
    execute = Mock(side_effect=RuntimeError("exec requested"))
    monkeypatch.setattr(worker, "validate_source", check)
    monkeypatch.setattr(worker.os, "execve", execute)
    monkeypatch.chdir(worker.REPO)
    return path, manifest, check, execute


@pytest.mark.parametrize("task,seed", [(1, 0), (20, 0), (21, 1), (40, 1)])
def test_executes_exact_one_based_row_and_records_receipt(*, execution, task, seed):
    path, manifest, check, execute = execution
    row = manifest["runs"][task - 1]
    with pytest.raises(RuntimeError, match="exec requested"):
        worker.run_worker(
            manifest_path=path, sha=manifest["source"]["head"], task=task, size="pilot"
        )
    executable, argv, environment = execute.call_args.args
    assert executable == sys.executable
    assert argv == [sys.executable, "-m", "hitl_pmp.cli", *row["argv"]]
    assert row["seed"] == seed
    assert environment["PYTHONPATH"].split(":")[:2] == [str(worker.REPO / "src"), str(worker.REPO)]
    receipt = json.loads((Path(row["output_dir"]) / "fidelity_launch.json").read_text())
    assert receipt["row"] == row
    assert receipt["task"] == task
    assert receipt["command"] == argv
    assert receipt["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt["source"] == check.return_value
    assert receipt["status"] == "validated; execution requested, not completion evidence"
    assert check.call_count == 2
    assert sum(Path(item["output_dir"]).exists() for item in manifest["runs"]) == 1


def test_worker_preserves_both_pinned_simulator_import_paths(*, execution, monkeypatch):
    path, manifest, _, execute = execution
    sources = {
        worker.REPO / "reference" / "kindergarden" / "src",
        worker.REPO / "reference" / "kinder-baselines" / "kinder-models" / "src",
    }
    original_is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda self: self in sources or original_is_dir(self))
    with pytest.raises(RuntimeError, match="exec requested"):
        worker.run_worker(manifest_path=path, sha=manifest["source"]["head"], task=1, size="pilot")
    environment = execute.call_args.args[2]
    assert {str(source) for source in sources} <= set(environment["PYTHONPATH"].split(":"))


def test_full_last_row_and_receipt_are_accepted_by_real_cli_validation(*, execution):
    from hitl_pmp.cli import Cli
    from hitl_pmp.step_protocol import StepProtocolValidation

    path, pilot, _, execute = execution
    manifest = build_manifest(
        size="full", goal_horizon=5, results_root=Path(pilot["options"]["results_root"])
    )
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="exec requested"):
        worker.main(
            argv=[
                "worker",
                "--manifest",
                str(path),
                "--sha",
                manifest["source"]["head"],
                "--task",
                "60",
                "--size",
                "full",
            ]
        )
    row = manifest["runs"][59]
    assert row["seed"] == 2
    assert execute.call_args.args[1][3:] == row["argv"]
    StepProtocolValidation.check(args=Cli.parse_args(argv=row["argv"]))


def test_dirty_source_refused_before_manifest_is_opened(*, monkeypatch, tmp_path):
    check = Mock(side_effect=ValueError("dirty tracked files"))
    execute = Mock()
    monkeypatch.setattr(worker, "validate_source", check)
    monkeypatch.setattr(worker.os, "execve", execute)
    with pytest.raises(ValueError, match="dirty tracked"):
        worker.run_worker(
            manifest_path=tmp_path / "missing.json", sha="a" * 40, task=1, size="pilot"
        )
    execute.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_foreign_editable_import_refused(*, execution, monkeypatch):
    path, manifest, _, execute = execution
    monkeypatch.setattr(sys.modules["hitl_pmp.cli"], "__file__", "/wrong/src/hitl_pmp/cli.py")
    with pytest.raises(ValueError, match="imported from the wrong checkout"):
        worker.run_worker(manifest_path=path, sha=manifest["source"]["head"], task=1, size="pilot")
    execute.assert_not_called()
    assert not Path(manifest["options"]["results_root"]).exists()


@pytest.mark.parametrize("task", [0, -1, 41, True, 1.5])
def test_invalid_task_refused_without_output(*, execution, task):
    path, manifest, _, execute = execution
    with pytest.raises(ValueError, match="1-based"):
        worker.run_worker(
            manifest_path=path, sha=manifest["source"]["head"], task=task, size="pilot"
        )
    execute.assert_not_called()
    assert not Path(manifest["options"]["results_root"]).exists()


@pytest.mark.parametrize("kind", ["directory", "file", "dangling_symlink"])
def test_existing_output_is_never_reused(*, execution, kind):
    path, manifest, _, execute = execution
    output = Path(manifest["runs"][0]["output_dir"])
    output.parent.mkdir(parents=True)
    if kind == "directory":
        output.mkdir()
    elif kind == "file":
        output.write_text("keep me")
    else:
        output.symlink_to(output.parent / "missing")
    with pytest.raises(FileExistsError):
        worker.run_worker(manifest_path=path, sha=manifest["source"]["head"], task=1, size="pilot")
    execute.assert_not_called()
    assert not (output / "fidelity_launch.json").exists()


@pytest.mark.parametrize("change", ["argv", "order", "config", "checkout", "head", "size"])
def test_manifest_tampering_fails_before_output(*, execution, change):
    path, manifest, _, execute = execution
    altered = copy.deepcopy(manifest)
    size = "pilot"
    if change == "argv":
        altered["runs"][0]["argv"].extend(["--seed", "999"])
    elif change == "order":
        altered["runs"].reverse()
    elif change == "config":
        altered["runs"][0]["config"]["seed"] = 999
    elif change == "size":
        size = "full"
    else:
        altered["source"][change] = "wrong"
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError):
        worker.run_worker(manifest_path=path, sha=manifest["source"]["head"], task=1, size=size)
    execute.assert_not_called()
    assert not Path(manifest["options"]["results_root"]).exists()


def test_source_change_during_validation_fails_before_output(*, execution):
    path, manifest, check, execute = execution
    check.side_effect = [check.return_value, ValueError("tracked source changed")]
    with pytest.raises(ValueError, match="tracked source changed"):
        worker.run_worker(manifest_path=path, sha=manifest["source"]["head"], task=1, size="pilot")
    execute.assert_not_called()
    assert not Path(manifest["options"]["results_root"]).exists()


@pytest.mark.parametrize("sha", ["HEAD", "a" * 7, "a" * 39, "g" * 40, "a" * 40 + "\n"])
def test_requires_full_literal_commit(*, monkeypatch, sha):
    git = Mock()
    monkeypatch.setattr(worker.subprocess, "check_output", git)
    with pytest.raises(ValueError, match="40-character"):
        worker.validate_source(sha=sha)
    git.assert_not_called()


@pytest.mark.parametrize("failure", [None, "head", "root", "dirty", "untracked_worker"])
def test_source_attestation_checks_head_root_index_and_worktree(*, monkeypatch, failure):
    sha = "a" * 40

    def git_output(*args, cwd, text):
        command = args[0]
        assert cwd == worker.REPO
        assert text is True
        assert command[0] == "git"
        if command[1:] == ["rev-parse", "--show-toplevel"]:
            return str(worker.REPO if failure != "root" else Path("/wrong")) + "\n"
        if command[1:] == ["rev-parse", "HEAD"]:
            return ("c" * 40 if failure == "head" else sha) + "\n"
        if command[1] == "status":
            assert "--untracked-files=no" in command
            assert "--ignore-submodules=none" in command
            return " M src/hitl_pmp/cli.py\n" if failure == "dirty" else ""
        if command[1] == "ls-files":
            assert "--error-unmatch" in command
            assert "scripts/tossing3d_fidelity_worker.py" in command
            if failure == "untracked_worker":
                raise subprocess.CalledProcessError(1, command)
            return "tracked\n"
        assert command[1:] == ["rev-parse", "HEAD^{tree}"]
        return "b" * 40 + "\n"

    monkeypatch.setattr(worker.subprocess, "check_output", git_output)
    if failure:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            worker.validate_source(sha=sha)
    else:
        assert worker.validate_source(sha=sha) == {
            "head": sha,
            "tree": "b" * 40,
            "clean_tracked": True,
        }


@pytest.mark.parametrize("size,tasks", [("pilot", 40), ("full", 60)])
def test_sbatch_is_uncapped_and_does_not_submit(*, size, tasks, monkeypatch, capsys):
    execute = Mock()
    monkeypatch.setattr(worker.os, "execve", execute)
    assert (
        worker.main(
            argv=[
                "sbatch-template",
                "--size",
                size,
                "--sha",
                "a" * 40,
                "--tag",
                "fidelity",
                "--walltime",
                "01:02:00",
            ]
        )
        == 0
    )
    template = capsys.readouterr().out
    assert f"#SBATCH --array=1-{tasks}\n" in template
    assert "#SBATCH --time=01:02:00\n" in template
    assert "--dependency" not in template
    assert "scripts.tossing3d_fidelity_worker worker" in template
    assert '--manifest "$1" --task "$SLURM_ARRAY_TASK_ID"' in template
    assert f"--size {size}" in template
    assert "FD_EXEC_PATH=/scratch/gpfs/TSILVER/jr2860/downward" in template
    assert "with_env.sh" in template
    execute.assert_not_called()
    subprocess.run(["bash", "-n"], input=template, text=True, check=True)


@pytest.mark.parametrize("walltime", ["01:01:59", "00:10:00", "1:99:00", "62", "\n48:00:00"])
def test_template_refuses_short_or_invalid_walltime(*, walltime):
    with pytest.raises(ValueError, match="walltime"):
        worker.sbatch_template(size="full", sha="a" * 40, tag="safe", walltime=walltime)


@pytest.mark.parametrize("tag", ["bad\n#SBATCH --array=1-2", "$(touch bad)", "bad name"])
def test_template_rejects_shell_or_directive_injection(*, tag):
    with pytest.raises(ValueError, match="tag"):
        worker.sbatch_template(size="pilot", sha="a" * 40, tag=tag, walltime="48:00:00")
