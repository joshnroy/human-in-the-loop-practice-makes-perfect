"""Lease, approval and persistent-job boundaries for actual AI continuation."""

import datetime as dt
import importlib.util
import json
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location(
    "wakeup", Path(__file__).parents[2] / "scripts/sweep_ai_wakeup.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def watcher(*, tmp_path, monkeypatch):
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Preserve pending approval; continue authorized work only.")
    owner = tmp_path / "owner.json"
    owner.write_text(
        json.dumps({
            "owner": "agent",
            "status": "active",
            "worktree": str(worktree),
            "last_progress_utc": dt.datetime.fromtimestamp(
                time.time() - 8000, dt.timezone.utc
            ).isoformat(),
            "current_units": [],
        })
    )
    task = {
        "name": "drawer",
        "owner_status": str(owner),
        "worktree": str(worktree),
        "prompt_file": str(prompt),
    }
    instance = module.Wakeup(config={"tasks": [task]}, state_dir=tmp_path / "state")
    monkeypatch.setattr(instance, "execute", Mock(return_value="complete"))
    monkeypatch.setattr(instance, "unit_active", Mock(return_value=False))
    return instance, owner


def update(*, path, **changes):
    value = json.loads(path.read_text())
    value.update(changes)
    path.write_text(json.dumps(value))


def test_stale_owner_runs_real_continuation_command(*, watcher):
    instance, owner = watcher
    assert instance.tick()[0]["status"] == "complete"
    call = instance.execute.call_args.kwargs
    assert "--approve-for-me" in call["argv"]
    assert "--sandbox" not in call["argv"]  # --approve-for-me implies workspace-write
    assert not any("bypass" in x for x in call["argv"])
    assert "Continue unfinished authorized implementation" in call["prompt"]
    assert json.loads(owner.read_text())["status"] == "idle-continuation-returned"


def test_fresh_owner_never_interrupted(*, watcher):
    instance, owner = watcher
    update(
        path=owner,
        lease_expires_utc=dt.datetime.fromtimestamp(
            time.time() + 3000, dt.timezone.utc
        ).isoformat(),
    )
    assert instance.tick()[0]["status"] == "owner-active"
    instance.execute.assert_not_called()


def test_pending_design_approval_never_bypassed(*, watcher):
    instance, owner = watcher
    update(path=owner, status="blocked-pending-third-native-approval")
    assert instance.tick()[0]["status"] == "waiting-authorized-next-step"
    instance.execute.assert_not_called()


def test_active_fresh_validation_not_relaunched(*, watcher):
    instance, owner = watcher
    update(
        path=owner,
        current_units=["validation.service"],
        last_progress_utc=dt.datetime.fromtimestamp(
            time.time() - 3000, dt.timezone.utc
        ).isoformat(),
    )
    instance.unit_active.return_value = True
    assert instance.tick()[0]["status"] == "validation-running"
    instance.execute.assert_not_called()


def test_stalled_validation_diagnosis_read_only(*, watcher):
    instance, owner = watcher
    update(path=owner, current_units=["validation.service"])
    instance.unit_active.return_value = True
    before = owner.read_text()
    assert instance.tick()[0]["status"] == "complete"
    argv = instance.execute.call_args.kwargs["argv"]
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert "--approve-for-me" not in argv
    assert owner.read_text() == before


def test_worktree_mismatch_blocks(*, watcher):
    instance, owner = watcher
    update(path=owner, worktree="/tmp/not-authorized")
    assert instance.tick()[0]["status"] == "supervisor-error"
    instance.execute.assert_not_called()


def test_failure_logged_without_science_retry(*, watcher):
    instance, owner = watcher
    instance.execute.return_value = "exit-1"
    assert instance.tick()[0]["status"] == "exit-1"
    assert instance.execute.call_count == 1
    assert json.loads(owner.read_text())["last_cli_outcome"] == "exit-1"


def test_lock_prevents_second_agent(*, watcher):
    import fcntl

    instance, _ = watcher
    with (instance.state_dir / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert instance.tick()["status"] == "already-running"
    instance.execute.assert_not_called()
