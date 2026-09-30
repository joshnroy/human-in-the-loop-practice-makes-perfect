"""Prepare reviewable Simple jobs; never publish READY or launch processes."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from hitl_pmp.environments.sweep_simple3d.cli import SweepSimpleCli


class SimpleReadiness:
    GATES = (
        "native_forward", "native_recovery", "native_goal_parity", "parameter_support",
        "action_accounting", "calibration_pick", "calibration_sweep", "ees_smoke", "pomdp_smoke",
    )

    @staticmethod
    def digest(*, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def prepare(cls, *, manifest_path: Path, evidence_path: Path,
                source: Path, output_root: Path) -> dict:
        manifest = json.loads(manifest_path.read_text())
        SweepSimpleCli.validate_manifest(manifest=manifest)
        revision = manifest.get("source_revision", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Requires a full frozen source_revision")
        evidence = json.loads(evidence_path.read_text())
        if set(evidence.get("checks", {})) != set(cls.GATES):
            raise ValueError("Missing or unexpected readiness evidence gates")
        checks = {}
        for name in cls.GATES:
            item = evidence["checks"][name]
            path = Path(item["path"])
            if not path.is_absolute():
                path = evidence_path.parent / path
            if not path.is_file() or cls.digest(path=path) != item.get("sha256"):
                raise ValueError(f"Missing or changed evidence: {name}")
            record = json.loads(path.read_text())
            if record.get("status") != "PASS" or record.get("revision") != revision:
                raise ValueError(f"Failing or stale evidence: {name}")
            checks[name] = {"path": str(path.resolve()), "sha256": item["sha256"]}
        # This is an evidence-summary scaffold, not an interpretation of raw
        # replay success or a replacement for physical/calibration gate audits.
        source_files = evidence.get("source_files", {})
        required = {"scripts/with_sweep_simple_env.sh", "scripts/run_sweep_verified_arm.py",
                    "scripts/run_sweep_manifest_arm.py"}
        if not required <= set(source_files):
            raise ValueError("Missing verified launcher source hashes")
        for name, digest in source_files.items():
            path = (source / name).resolve()
            if not path.is_relative_to(source.resolve()) or cls.digest(path=path) != digest:
                raise ValueError(f"Source evidence mismatch: {name}")
        jobs = []
        arms = {arm["name"]: arm for arm in manifest["arms"]}
        order = manifest["launch_order"]
        flattened = [name for stage in order for name in stage]
        if len(flattened) != len(set(flattened)) or set(flattened) != set(arms):
            raise ValueError("Launch order must include each arm exactly once")
        first = manifest["first_seed"]
        if first not in manifest["valid_practice_seeds"]:
            raise ValueError("First seed is not a validated practice seed")
        seeds = [first] + [s for s in manifest["valid_practice_seeds"] if s != first]
        for stage, names in enumerate(order):
            for name in names:
                if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
                    raise ValueError("Unsafe arm identifier")
                for seed_index, seed in enumerate(seeds):
                    job_id = f"{name}-s{seed}"
                    output = output_root / name / f"seed-{seed}"
                    completion = output / "completion.json"
                    jobs.append(dict(
                        id=job_id, arm=name, seed=seed, stage=stage + seed_index * len(order),
                        output=str(output.resolve()),
                        progress_path=str((output / arms[name]["method"] / str(seed)
                                           / "progress.jsonl").resolve()),
                        completion_record=str(completion.resolve()),
                        argv=[str((source / "scripts/with_sweep_simple_env.sh").resolve()),
                              "python",
                              str((source / "scripts/run_sweep_verified_arm.py").resolve()),
                              "--manifest", str(manifest_path.resolve()), "--arm", name,
                              "--seed", str(seed), "--output", str(output.resolve()),
                              "--completion-record", str(completion.resolve()),
                              "--job-id", job_id, "--revision", revision],
                    ))
        return dict(
            status="DRAFT", environment="sweep_simple3d", owner_validated=False,
            purpose="Prepared jobs only; owner must audit evidence before readiness publication",
            source=str(source.resolve()), revision=revision,
            manifest=str(manifest_path.resolve()), manifest_sha256=cls.digest(path=manifest_path),
            checks=checks, source_files=source_files,
            memory_max_bytes=6 * 1024**3, memory_swap_max_bytes=0,
            output_root=str(output_root.resolve()), jobs=jobs,
        )

    @classmethod
    def main(cls) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        for name in ("manifest", "evidence", "source", "output-root", "output"):
            parser.add_argument("--" + name, type=Path, required=True)
        args = parser.parse_args()
        prepared = cls.prepare(manifest_path=args.manifest, evidence_path=args.evidence,
                               source=args.source, output_root=args.output_root)
        with args.output.open("x") as stream:
            json.dump(prepared, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    SimpleReadiness.main()
