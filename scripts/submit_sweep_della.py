"""Prepare pinned isolated Sweep sources, then submit exactly one Slurm array.

Idempotence and ambiguous-submission recovery belong to the external persistent
readiness watcher. This wrapper never retries sbatch.
"""

# ruff: noqa: E501 (generated batch shell statements)
import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

SCRATCH = "/scratch/gpfs/TSILVER/jr2860"


def ssh(command):  # noqa: PLR0917 (small shell transport helper)
    wrapped = (
        "systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 -p OOMPolicy=continue bash -c "
        + shlex.quote(command)
    )
    return subprocess.check_output(["ssh", "-o", "BatchMode=yes", "della", wrapped], text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{20}", args.token):
        raise ValueError("Expected watcher-generated20hex token")
    root = Path(__file__).resolve().parent.parent
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("source_revision") != revision:
        raise ValueError("Manifest source revision differs from submitting source")
    if manifest.get("status") != "READY":
        raise ValueError("Scientific manifest is not ready")
    remote = f"{SCRATCH}/code/{revision}"
    batch = f"{SCRATCH}/sweep-launch/{args.token}"
    # Only Josh's own fork is fetched; never change installed dependency pins.
    commands = [
        "set -e",
        f"git -C {SCRATCH}/hitl-pmp fetch -q https://github.com/joshnroy/human-in-the-loop-practice-makes-perfect.git {revision}",
        f"test -d {remote} || git -C {SCRATCH}/hitl-pmp worktree add --detach {remote} {revision}",
        f"mkdir -p {batch} {SCRATCH}/slurm-logs",
    ]
    for name in ("kindergarden", "kinder-baselines"):
        pin = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", f"HEAD:reference/{name}"], text=True
        ).strip()
        target = f"{remote}/reference/{name}"
        commands += [
            f"test -f {target}/.git || test -d {target}/.git || git clone -q --shared --no-checkout {SCRATCH}/hitl-pmp/reference/{name} {target}",
            f"git -C {target} checkout -q --detach {pin}",
            f"test $(git -C {target} rev-parse HEAD) = {pin}",
        ]
    assets = "reference/kindergarden/src/kinder/envs/dynamic3d/models/assets/mimiclabs_scenes/"
    commands += [f"rsync -a --ignore-existing {SCRATCH}/hitl-pmp/{assets} {remote}/{assets}"]
    print(ssh("\n".join(commands)), file=sys.stderr)
    destination = f"{batch}/manifest.json"
    subprocess.run(
        ["scp", "-q", "-o", "BatchMode=yes", str(args.manifest), f"della:{destination}"], check=True
    )
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if ssh(f"sha256sum {destination}").split()[0] != digest:
        raise RuntimeError("Remote manifest checksum mismatch")
    # One array encompasses the approved first cheap/high2x2. No hidden seed expansion.
    arms = manifest["launch_order"][0]
    if len(arms) != 4:
        raise ValueError("First batch must contain exactly four approved arms")
    script = f"""#!/bin/bash
#SBATCH --account=tsilver
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=08:00:00
#SBATCH --output={SCRATCH}/slurm-logs/%x-%A_%a.out
set -euo pipefail
export CONDA_SH=/usr/licensed/anaconda3/2026.7.1/etc/profile.d/conda.sh
export HITL_PMP_CONDA_ENV={SCRATCH}/envs/hitl-pmp
export FD_EXEC_PATH={SCRATCH}/downward
export DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1
export XDG_RUNTIME_DIR="${{TMPDIR:-/tmp}}/xdg-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_RUNTIME_DIR"
export LD_PRELOAD="$HITL_PMP_CONDA_ENV/lib/libstdc++.so.6"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd {remote}
{shlex.quote(str(SCRATCH))}/envs/hitl-pmp/bin/python scripts/check_slurm_memory.py
arms=({" ".join(shlex.quote(a) for a in arms)})
arm="${{arms[$SLURM_ARRAY_TASK_ID]}}"
exec scripts/with_sweep_env.sh python scripts/run_sweep_manifest_arm.py --manifest {destination} --arm "$arm" --seed {manifest["first_seed"]} --output {SCRATCH}/results/sweep-faithful/{args.token}/"$arm"
"""
    local = args.manifest.parent / f"array-{args.token}.sbatch"
    local.write_text(script)
    subprocess.run(
        ["scp", "-q", "-o", "BatchMode=yes", str(local), f"della:{batch}/array.sbatch"], check=True
    )
    if args.prepare_only:
        print(json.dumps({"prepared": batch, "revision": revision}))
        return
    result = ssh(
        f"sbatch --parsable --array=0-3 --job-name=sweep-{args.token} {batch}/array.sbatch"
    ).strip()
    job = result.split(";")[0]
    if not job.isdigit():
        raise RuntimeError(f"Ambiguous sbatch response: {result!r}")
    print(json.dumps({"job_ids": [job], "revision": revision, "remote_manifest": destination}))


if __name__ == "__main__":
    main()
