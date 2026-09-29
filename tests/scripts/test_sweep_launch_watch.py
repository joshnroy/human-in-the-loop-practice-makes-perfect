"""Exercise submission acknowledgement loss and restart safety."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

SPEC = importlib.util.spec_from_file_location(
    "watch", Path(__file__).parents[2] / "scripts/sweep_launch_watch.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def watcher(*, tmp_path):
    readiness = tmp_path / "readiness.json"
    artifact = tmp_path / "evidence.json"
    artifact.write_text("{}")
    watch = MODULE.LaunchWatch(readiness=readiness, state_dir=tmp_path / "state")
    readiness.write_text(
        json.dumps(
            dict(
                status="READY",
                revision="a" * 40,
                source_dir=str(tmp_path),
                manifest=dict(path=str(artifact), sha256=watch.digest(path=artifact)),
                evidence=[dict(path=str(artifact), sha256=watch.digest(path=artifact))],
                submit_argv=["submit", "{token}"],
            )
        )
    )
    watch.reconcile = Mock(return_value=[])
    watch.run = Mock(side_effect=["a" * 40, "", '{"job_ids":["1234"]}'])
    return watch


def test_submit_once_across_ticks(*, watcher):
    watcher.tick()
    watcher.tick()
    assert watcher.state["job_ids"] == ["1234"]
    assert watcher.run.call_count == 3


def test_existing_batch_not_submitted(*, watcher):
    watcher.reconcile.return_value = ["987"]
    watcher.tick()
    assert watcher.state["job_ids"] == ["987"]
    assert watcher.run.call_count == 2


def test_lost_ack_reconciles_without_retry(*, watcher):
    watcher.run.side_effect = ["a" * 40, "", TimeoutError("lost SSH")]
    watcher.tick()
    assert watcher.state["submission_intent"]
    watcher.reconcile.return_value = ["555"]
    watcher.tick()
    assert watcher.state["job_ids"] == ["555"]
    assert watcher.run.call_count == 3


def test_lost_ack_empty_accounting_does_not_retry(*, watcher):
    watcher.run.side_effect = ["a" * 40, "", TimeoutError("lost SSH")]
    watcher.tick()
    watcher.tick()
    assert watcher.state["status"] == "blocked"
    assert watcher.run.call_count == 3


def test_no_readiness_waits(*, watcher):
    watcher.readiness.unlink()
    watcher.tick()
    assert watcher.state["status"] == "waiting"
    watcher.run.assert_not_called()


def test_tampered_evidence_blocks(*, watcher):
    Path(json.loads(watcher.readiness.read_text())["manifest"]["path"]).write_text("changed")
    watcher.tick()
    assert watcher.state["status"] == "blocked"
    assert not watcher.state.get("submission_intent")


def test_dirty_source_blocks(*, watcher):
    watcher.run.side_effect = ["a" * 40, " M controller.py"]
    watcher.tick()
    assert "dirty" in watcher.state["reason"]
    assert not watcher.state.get("submission_intent")


def test_ssh_failure_before_intent_can_retry(*, watcher):
    watcher.reconcile.side_effect = TimeoutError("SSH unavailable")
    watcher.tick()
    assert not watcher.state.get("submission_intent")
    assert watcher.state["status"] == "blocked"


def test_restart_retains_intent(*, watcher):
    watcher.run.side_effect = ["a" * 40, "", TimeoutError("lost SSH")]
    watcher.tick()
    restarted = MODULE.LaunchWatch(readiness=watcher.readiness, state_dir=watcher.root)
    restarted.reconcile = Mock(return_value=[])
    restarted.run = Mock()
    restarted.tick()
    restarted.run.assert_not_called()
    assert restarted.state["submission_intent"]


def test_scheduler_matching_uses_exact_name(*, watcher):
    watcher.run = Mock(
        return_value="123_0|sweep-abc\n123_1|sweep-abc\n999|sweep-abcdef\n123.batch|batch"
    )
    assert MODULE.LaunchWatch.reconcile(watcher, token="abc") == ["123"]


def test_readiness_not_ready_blocks(*, watcher):
    watcher.readiness.write_text('{"status":"BLOCKED","reason":"physical check pending"}')
    watcher.tick()
    assert watcher.state["reason"] == "physical check pending"
    watcher.run.assert_not_called()


def test_wrong_revision_blocks(*, watcher):
    watcher.run.side_effect = ["b" * 40]
    watcher.tick()
    assert watcher.state["reason"] == "Source revision changed"
    assert not watcher.state.get("submission_intent")


def test_lock_prevents_concurrent_submission(*, watcher):
    import fcntl

    with (watcher.root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        watcher.tick()
    watcher.run.assert_not_called()
