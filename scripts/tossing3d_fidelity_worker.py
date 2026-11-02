"""Execute one validated fidelity row, or print a Della template; never submit jobs.

Prepare the JSON manifest on the final deployed checkout using
``scripts.tossing3d_ees_fidelity_launch prepare``. Its checkout-specific paths and
source attestation must match exactly; this worker does not relocate manifests.
``sbatch-template`` prints an uncapped array script taking that manifest as $1.
Neither template generation nor a launch receipt is pilot or full-gate evidence.
The deployment must remain immutable after validation, including dependencies,
assets and untracked files, which the tracked-source check does not attest.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DELLA = "/scratch/gpfs/TSILVER/jr2860"
TASK_COUNTS = {"pilot": 40, "full": 60}


def validate_sha(*, sha: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", sha) is None:
        raise ValueError("sha must be a full lowercase 40-character commit ID")


def validate_source(*, sha: str) -> dict:
    """Attest HEAD and both tracked index/worktree state, including submodules."""
    validate_sha(sha=sha)

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != REPO:
        raise ValueError("worker must belong to the deployed Git checkout")
    head = git("rev-parse", "HEAD")
    if head != sha:
        raise ValueError(f"deployed HEAD {head} does not match requested commit {sha}")
    if git("status", "--porcelain=v1", "--untracked-files=no", "--ignore-submodules=none"):
        raise ValueError("deployed checkout has dirty tracked files or submodules")
    git(
        "ls-files",
        "--error-unmatch",
        "--",
        "scripts/tossing3d_fidelity_worker.py",
        "scripts/tossing3d_ees_fidelity_launch.py",
        "src/hitl_pmp/cli.py",
    )
    return {"head": head, "tree": git("rev-parse", "HEAD^{tree}"), "clean_tracked": True}


def run_worker(*, manifest_path: Path, sha: str, task: int, size: str) -> None:
    """Fail closed before claiming output; replace this process with the real CLI."""
    validate_source(sha=sha)
    from scripts import tossing3d_ees_fidelity_launch as launcher

    if launcher.REPO.resolve() != REPO or Path(launcher.__file__).resolve().parents[1] != REPO:
        raise ValueError("manifest validator imported from the wrong checkout")
    for name, module in tuple(sys.modules.items()):
        if name == "hitl_pmp" or name.startswith("hitl_pmp."):
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(REPO / "src"):
                raise ValueError(f"{name} imported from the wrong checkout")
    manifest_path = manifest_path.resolve(strict=True)
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    if size not in TASK_COUNTS or manifest["options"]["size"] != size:
        raise ValueError("manifest size does not match the requested array")
    if manifest["source"]["head"] != sha:
        raise ValueError("manifest HEAD does not match the requested commit")
    launcher.validate_manifest(manifest=manifest)
    rows = manifest["runs"]
    if len(rows) != TASK_COUNTS[size]:
        raise ValueError("manifest row count does not match the requested array")
    if type(task) is not int or not 1 <= task <= len(rows):
        raise ValueError(f"task must be a 1-based integer in [1, {len(rows)}]")
    row = rows[task - 1]
    source = validate_source(sha=sha)
    command = [sys.executable, "-m", "hitl_pmp.cli", *row["argv"]]
    environment = dict(os.environ)
    python_paths = [str(REPO / "src"), str(REPO)]
    kinder_source = REPO / "reference" / "kindergarden" / "src"
    if kinder_source.is_dir():
        python_paths.append(str(kinder_source))
    models_source = REPO / "reference" / "kinder-baselines" / "kinder-models" / "src"
    if models_source.is_dir():
        python_paths.append(str(models_source))
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    receipt = {
        "schema_version": 1,
        "status": "validated; execution requested, not completion evidence",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "checkout": str(REPO),
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "manifest_source": manifest["source"],
        "task": task,
        "size": size,
        "row": row,
        "command": command,
        "environment": {
            name: environment.get(name)
            for name in (
                "PYTHONPATH",
                "FD_EXEC_PATH",
                "CONDA_PREFIX",
                "HITL_PMP_TMPDIR",
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
            )
        },
        "provenance_pending": manifest["provenance_pending"],
    }
    output = Path(row["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    with (output / "fidelity_launch.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chdir(REPO)
    os.execve(sys.executable, command, environment)


def sbatch_template(*, size: str, sha: str, tag: str, walltime: str) -> str:
    """Print-only scheduler wiring, with no gate, dependencies or concurrency cap."""
    validate_sha(sha=sha)
    if size not in TASK_COUNTS:
        raise ValueError("size must be pilot or full")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", tag) is None:
        raise ValueError("tag must contain only letters, digits, underscores or hyphens")
    matched = re.fullmatch(r"([0-9]{2,3}):([0-5][0-9]):([0-5][0-9])", walltime)
    if matched is None:
        raise ValueError("walltime must be HH:MM:SS and at least 01:02:00")
    hours, minutes, seconds = map(int, matched.groups())
    if hours * 3600 + minutes * 60 + seconds < 62 * 60:
        raise ValueError("walltime must be at least 01:02:00")
    return f"""#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time={walltime}
#SBATCH --array=1-{TASK_COUNTS[size]}
#SBATCH --job-name={tag}-{size}
#SBATCH --output={DELLA}/slurm-logs/%x-%A_%a.out
set -euo pipefail
CODE={DELLA}/code/{sha}
export CONDA_SH=/usr/licensed/anaconda3/2026.7.1/etc/profile.d/conda.sh
export HITL_PMP_CONDA_ENV={DELLA}/envs/hitl-pmp
export FD_EXEC_PATH={DELLA}/downward
export DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1
export HITL_PMP_TMPDIR={DELLA}/tmp/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID
mkdir -p "$HITL_PMP_TMPDIR"
export TMP="$HITL_PMP_TMPDIR" TMPDIR="$HITL_PMP_TMPDIR" TEMP="$HITL_PMP_TMPDIR"
export XDG_RUNTIME_DIR="$(umask 077; mktemp -d /tmp/t3-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID.XXXXXX)"
chmod 700 "$XDG_RUNTIME_DIR"
export LD_PRELOAD="$HITL_PMP_CONDA_ENV/lib/libstdc++.so.6"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$CODE"
exec "$CODE/scripts/with_env.sh" python -m scripts.tossing3d_fidelity_worker worker \\
  --sha {sha} --size {size} --manifest "$1" --task "$SLURM_ARRAY_TASK_ID"
"""


def main(*, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    worker = modes.add_parser("worker")
    worker.add_argument("--sha", required=True)
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--task", type=int, required=True)
    worker.add_argument("--size", choices=TASK_COUNTS, required=True)
    template = modes.add_parser("sbatch-template")
    template.add_argument("--sha", required=True)
    template.add_argument("--size", choices=TASK_COUNTS, required=True)
    template.add_argument("--tag", required=True)
    template.add_argument("--walltime", required=True, help="HH:MM:SS, at least 01:02:00")
    args = parser.parse_args(argv)
    if args.mode == "worker":
        run_worker(manifest_path=args.manifest, sha=args.sha, task=args.task, size=args.size)
    else:
        print(sbatch_template(size=args.size, sha=args.sha, tag=args.tag, walltime=args.walltime))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
