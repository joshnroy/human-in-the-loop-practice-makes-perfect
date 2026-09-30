"""Readiness scaffolding cannot turn missing or failing evidence into READY."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("failure", [None, "missing", "failed", "stale", "hash"])
def test_evidence_and_verified_wrapper_jobs(*, tmp_path, monkeypatch, failure):
    script = Path(__file__).resolve().parents[2] / "scripts/prepare_sweepsimple_readiness.py"
    spec = importlib.util.spec_from_file_location("simple_readiness_builder_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = module.SimpleReadiness
    # Protocol validation has separate tests; isolate evidence/wrapper assembly here.
    monkeypatch.setattr(module.SweepSimpleCli, "validate_manifest", lambda **kwargs: None)
    revision = "a" * 40
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(
        source_revision=revision, first_seed=0, valid_practice_seeds=[0, 1],
        arms=[dict(name="ees-low", method="ees"), dict(name="pomdp-low", method="pomdp")],
        launch_order=[["ees-low", "pomdp-low"]],
    )))
    evidence = {"checks": {}, "source_files": {}}
    for name in builder.GATES:
        path = tmp_path / f"{name}.json"
        record = dict(status="PASS", revision=revision)
        if name == "native_forward" and failure == "failed":
            record["status"] = "FAIL"
        if name == "native_forward" and failure == "stale":
            record["revision"] = "b" * 40
        path.write_text(json.dumps(record))
        evidence["checks"][name] = dict(path=str(path), sha256=builder.digest(path=path))
    if failure == "missing":
        del evidence["checks"]["native_forward"]
    if failure == "hash":
        (tmp_path / "native_forward.json").write_text("{}")
    for name in (
        "with_sweep_simple_env.sh", "run_sweep_verified_arm.py", "run_sweep_manifest_arm.py"
    ):
        relative = "scripts/" + name
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text("test fixture, not executable science source")
        evidence["source_files"][relative] = builder.digest(path=path)
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    if failure:
        with pytest.raises(ValueError):
            builder.prepare(manifest_path=manifest, evidence_path=evidence_path,
                            source=tmp_path, output_root=tmp_path / "results")
        return
    result = builder.prepare(manifest_path=manifest, evidence_path=evidence_path,
                             source=tmp_path, output_root=tmp_path / "results")
    assert result["status"] == "DRAFT" and result["owner_validated"] is False
    assert result["memory_swap_max_bytes"] == 0
    assert len(result["jobs"]) == 4
    for job in result["jobs"]:
        assert job["stage"] == job["seed"]
        argv = job["argv"]
        assert argv[2].endswith("run_sweep_verified_arm.py")
        assert argv[argv.index("--completion-record") + 1] == job["completion_record"]
        assert argv[argv.index("--revision") + 1] == revision
