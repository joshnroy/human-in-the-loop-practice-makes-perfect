"""Launch-time equivalence guard: the smoke must decide what a recorded reference decided.

Mandatory from v11: every package either binds a reference or records a waiver. A launch
that claims to be behaviour-neutral (a speedup, a cache) binds, at prepare time, each
smoke row to the run of a previous launch with the same reset cost, arm and seed and
the same argv (`prepare.py --equivalence-reference`). After the smoke, the gate runs

    python -m scripts.kitchen_recycling.equivalence --launch "$ROOT/launch"

which compares, over the smoke's own step range, its practice step events, its decision
log and the held-out evaluations it ran with the reference's, and writes
`equivalence-check.json` only if every one matches. Any divergence exits non-zero with the
first divergent record, so an `afterok` array never starts. A launch meant to change
behaviour instead records a waiver (`--equivalence-reference none --equivalence-waiver`),
and the check records that waiver. Either record must reach the full authorization's
provenance (`worker.check_equivalence`).

**The comparable range.** The smoke's budget interrupts the skill running at 3,200 steps
and forces a learning update; the reference practises on. So step events are compared up
to the smoke's first budget-affected event, decision records through the smoke's last
dispatch (that action was chosen before the budget hit), and evaluations at every
checkpoint the smoke measured before then, task by task (the smoke runs only the first N).

**Normalised away**, and nothing else: wall-clock fields (`timestamp`, `*_seconds`), the
policy *file* hash (a pickle's bytes embed its output path, `PolicyFileHash`), and, as in
#675's `compare.py`, memory addresses and the hash order of frozensets.

**Same CPU or refuse.** Search results are not bit-reproducible across CPU types (#675).
When both runs have `host.json` (written by `worker.run`), they must name the same CPU
model. A reference from before `host.json` existed (kitchen v10 and earlier) recorded no
model, so its identity is its Slurm node's feature instead: its array task's job id from
its `logs/*-full-<job>_<row>.out`, its node from `sacct`, that node's features from
`sinfo`, and both nodes must carry `genoa` (Josh, 2026-11-04: the node label is what we
control). The record names which identity was used.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from hitl_pmp.results_writer.jsonl_log import JsonlLog
from scripts.kitchen_recycling.worker import binding, output, read, sha

RECORD = "equivalence-check.json"
BUDGET = "practice_step_budget"
# A deployment snapshot's file hash: its pickle embeds the run's own output path.
FILE_IDENTITY = ("policy_file_sha256", "policy_sha256")
DECISION_LOGS = ("pomdp_decisions.jsonl", "ees_decisions.jsonl")
ADDRESS = re.compile(r" at 0x[0-9a-f]+")
# Lists serialised from frozensets: their order follows element hashes, and a
# predicate's `holds` lambda hashes by address, so it varies from process to process.
SET_LIKE = (
    "add_effects",
    "delete_effects",
    "ignore_effects",
    "preconditions",
    "conditions",
    "atoms",
    "before_atoms",
    "after_atoms",
    "conditional_add_effects",
    "conditional_ignore_effects",
    "true_atoms",
    "start_atoms",
)
SHOWN = 600
# The Slurm feature naming Della's AMD EPYC Genoa nodes: a legacy reference's CPU identity.
CPU_FAMILY_FEATURE = "genoa"


class Divergence(ValueError):
    """The smoke decided differently from its reference, or there was nothing to compare."""


class CpuMismatch(ValueError):
    """The two runs did not provably run on the same CPU model."""


def canon_key(*, key: str) -> str:
    """A `str(ground_skill)` key renders its frozensets in hash order (GroundSkillText),
    so it is compared as a multiset of characters, as #675's compare.py does."""
    key = ADDRESS.sub("", key)
    return "".join(sorted(key)) if key.startswith("skill=Skill(") else key


def is_wall_clock(*, key: str) -> bool:
    return key == "timestamp" or key.endswith("_seconds")


def scrub(*, value: object, key: str = "") -> object:
    """Drop wall-clock and file-identity fields; strip addresses; order set-like lists."""
    if isinstance(value, dict):
        return {
            canon_key(key=str(k)): scrub(value=v, key=str(k))
            for k, v in value.items()
            if not is_wall_clock(key=str(k)) and k not in FILE_IDENTITY
        }
    if isinstance(value, list):
        items = [scrub(value=v) for v in value]
        if key in SET_LIKE:
            items.sort(key=lambda v: json.dumps(v, sort_keys=True))
        return items
    if isinstance(value, str):
        return canon_key(key=value)
    return value


def records(*, path: Path) -> list[dict]:
    with JsonlLog.open_text(path=path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def first_difference(*, smoke: object, reference: object, where: str = "") -> str:
    """The key path of the first differing leaf, for the failure message."""
    if isinstance(smoke, dict) and isinstance(reference, dict):
        for key in sorted(set(smoke) | set(reference), key=str):
            if key not in smoke or key not in reference:
                return f"{where}.{key} (present in one only)"
            if smoke[key] != reference[key]:
                return first_difference(
                    smoke=smoke[key], reference=reference[key], where=f"{where}.{key}"
                )
    if isinstance(smoke, list) and isinstance(reference, list):
        for index, (left, right) in enumerate(zip(smoke, reference, strict=False)):
            if left != right:
                return first_difference(smoke=left, reference=right, where=f"{where}[{index}]")
        return f"{where} (length {len(smoke)} vs {len(reference)})"
    return where or "(whole record)"


def compare_records(*, name: str, smoke: list, reference: list) -> None:
    """Every smoke record equals the reference's at the same position."""
    for index, (left, right) in enumerate(zip(smoke, reference, strict=False)):
        left, right = scrub(value=left), scrub(value=right)
        if left != right:
            raise Divergence(
                f"{name}: first divergence at record {index}, "
                f"{first_difference(smoke=left, reference=right)}: "
                f"smoke {json.dumps(left, sort_keys=True)[:SHOWN]} vs reference "
                f"{json.dumps(right, sort_keys=True)[:SHOWN]}"
            )
    if len(reference) < len(smoke):
        raise Divergence(
            f"{name}: the reference ends after {len(reference)} records; "
            f"the smoke's comparable range has {len(smoke)}"
        )


def slurm_node(*, job: str) -> str:
    """The one node a finished Slurm job (array task) ran on, from accounting."""
    try:
        result = subprocess.run(
            ["sacct", "-j", job, "-X", "-n", "-P", "-o", "JobID,NodeList"],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CpuMismatch(f"sacct could not report job {job}: {error}") from error
    lines = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
    nodes = {line[1] for line in lines if len(line) == 2}
    if len(lines) != 1 or len(nodes) != 1 or any(c in next(iter(nodes)) for c in "[],"):
        raise CpuMismatch(f"sacct does not name exactly one node for job {job}: {lines}")
    return next(iter(nodes))


def node_features(*, node: str) -> set[str]:
    """A node's Slurm features (on Della, e.g. amd,genoa,rh9,nvme)."""
    try:
        result = subprocess.run(
            ["sinfo", "-h", "-N", "-n", node, "-o", "%f"],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise CpuMismatch(f"sinfo could not report node {node}: {error}") from error
    return {
        feature
        for line in result.stdout.splitlines()
        for feature in line.strip().split(",")
        if feature
    }


def reference_job(*, reference_root: Path, index: int) -> str:
    """A legacy reference row's array task, read from its full-array log's file name
    (`<tag>-full-<job>_<row>.out`); refused unless exactly one array ran that row."""
    pattern = re.compile(rf"-full-(\d+)_{index}\.out$")
    jobs = {
        match.group(1)
        for path in (reference_root / "logs").glob(f"*-full-*_{index}.out")
        if (match := pattern.search(path.name))
    }
    if len(jobs) != 1:
        raise CpuMismatch(
            f"{reference_root}/logs names {len(jobs)} full-array jobs for row {index}; "
            "exactly one is needed to look up the node it ran on"
        )
    return f"{next(iter(jobs))}_{index}"


def cpu_identity(*, smoke: Path, reference: Path, reference_job: str | None = None) -> dict:
    """How the two runs are known to share a CPU: the same `host.json` model, or, for a
    reference that predates `host.json`, both nodes carrying the `genoa` Slurm feature."""
    smoke_host = smoke / "host.json"
    if not smoke_host.is_file():
        raise CpuMismatch(f"{smoke_host} is missing: the smoke recorded no host")
    ours = read(path=smoke_host)
    if (reference / "host.json").is_file():
        theirs = read(path=reference / "host.json").get("cpu_model")
        if not ours.get("cpu_model") or ours.get("cpu_model") != theirs:
            raise CpuMismatch(
                f"the smoke ran on {ours.get('cpu_model')!r}, the reference on {theirs!r}"
            )
        return {"basis": "host.json cpu_model", "cpu_model": theirs}
    if reference_job is None:
        raise CpuMismatch(
            f"{reference}/host.json is missing and no Slurm job is known for the reference, "
            "so its CPU cannot be identified"
        )
    smoke_node = (
        ours.get("slurm", {}).get("SLURMD_NODENAME") or str(ours.get("hostname", "")).split(".")[0]
    )
    reference_node = slurm_node(job=reference_job)
    for name, node in (("smoke", smoke_node), ("reference", reference_node)):
        if CPU_FAMILY_FEATURE not in node_features(node=node):
            raise CpuMismatch(
                f"the {name} node {node} does not carry the {CPU_FAMILY_FEATURE!r} feature"
            )
    return {
        "basis": f"slurm-feature:{CPU_FAMILY_FEATURE}",
        "smoke_node": smoke_node,
        "reference_node": reference_node,
        "reference_job": reference_job,
    }


def budget_affected(*, event: dict) -> bool:
    return (
        event.get("event") == "practice_end"
        or event.get("interruption") == BUDGET
        or event.get("reason") == BUDGET
    )


def decision_log(*, run: Path) -> tuple[str, Path] | None:
    for name in DECISION_LOGS:
        if JsonlLog.exists(path=run / name):
            return name, JsonlLog.resolve(path=run / name)
    return None


def compare_evaluations(*, smoke: Path, reference: Path, checkpoints: list[int]) -> int:
    """Each smoke task at each checkpoint against the reference's task of that number."""
    ours = {entry["index"]: entry for entry in read(path=smoke / "step_evaluations.json")}
    theirs = {entry["index"]: entry for entry in read(path=reference / "step_evaluations.json")}
    compared = 0
    for checkpoint in checkpoints:
        if checkpoint not in ours or checkpoint not in theirs:
            raise Divergence(f"step_evaluations.json: checkpoint {checkpoint} missing")
        reference_tasks = {task["task"]: task for task in theirs[checkpoint]["tasks"]}
        for task in ours[checkpoint]["tasks"]:
            number = task["task"]
            name = f"step_evaluations.json: checkpoint {checkpoint} task {number}"
            if number not in reference_tasks:
                raise Divergence(f"{name}: the reference did not evaluate it")
            compare_records(name=name, smoke=[task], reference=[reference_tasks[number]])
            relative = Path("evaluations") / f"{checkpoint:04d}" / "tasks" / f"{number:02d}"
            for file in ("results.json", "actions.jsonl"):
                ours_path, theirs_path = smoke / relative / file, reference / relative / file
                if not ours_path.is_file():
                    continue
                if not theirs_path.is_file():
                    raise Divergence(f"{relative / file}: missing from the reference")
                load = (lambda path: [read(path=path)]) if file.endswith(".json") else None
                compare_records(
                    name=str(relative / file),
                    smoke=load(ours_path) if load else records(path=ours_path),
                    reference=load(theirs_path) if load else records(path=theirs_path),
                )
            compared += 1
    return compared


def compare_run(*, smoke: Path, reference: Path, reference_job: str | None = None) -> dict:
    """Refuse unless both ran on one CPU (`cpu_identity`); raise `Divergence` at the first
    record the smoke decided differently. Returns how much was compared, and how."""
    identity = cpu_identity(smoke=smoke, reference=reference, reference_job=reference_job)
    events = records(path=smoke / "step_events.jsonl")
    prefix = []
    for event in events:
        if budget_affected(event=event):
            break
        prefix.append(event)
    if not any(event.get("event") in {"robot", "human"} for event in prefix):
        raise Divergence("nothing to compare: the smoke took no action before its budget")
    compare_records(
        name="step_events.jsonl",
        smoke=prefix,
        reference=records(path=reference / "step_events.jsonl"),
    )
    logs = decision_log(run=smoke), decision_log(run=reference)
    if logs[0] is None or logs[1] is None or logs[0][0] != logs[1][0]:
        raise Divergence(f"no matching decision log: smoke {logs[0]}, reference {logs[1]}")
    name = logs[0][0]
    ours = records(path=logs[0][1])
    dispatches = [index for index, record in enumerate(ours) if record.get("event") == "dispatch"]
    if not dispatches:
        raise Divergence(f"nothing to compare: {name} records no dispatch")
    ours = ours[: dispatches[-1] + 1]
    compare_records(name=name, smoke=ours, reference=records(path=logs[1][1]))
    checkpoints = [event["index"] for event in prefix if event.get("event") == "measurement"]
    if not checkpoints:
        raise Divergence("nothing to compare: no evaluation checkpoint before the budget")
    tasks = compare_evaluations(smoke=smoke, reference=reference, checkpoints=checkpoints)
    return {
        "step_events": len(prefix),
        "decisions": {"file": name, "records": len(ours)},
        "evaluation_tasks": tasks,
        "cpu_identity": identity,
    }


def check(*, launch: Path) -> dict:
    """The gate's check: compare every bound smoke row, or record the waiver. Writes
    `equivalence-check.json` (exclusively) only when it passes or is waived."""
    manifest = read(path=launch / "manifest.json")
    block = manifest.get("equivalence")
    if block is None:
        raise ValueError(
            "The manifest binds no equivalence reference or waiver "
            "(prepare.py --equivalence-reference)"
        )
    bound = binding(launch=launch)
    if block["reference_root"] is None:
        record = {"binding": bound, "passed": False, "waived": True, "waiver": block["waiver"]}
    else:
        reference_manifest = Path(block["reference_root"]) / "launch" / "manifest.json"
        if sha(path=reference_manifest) != block["reference_manifest_sha256"]:
            raise ValueError(
                f"The reference manifest changed since prepare.py bound it: {reference_manifest}"
            )
        rows = {row["index"]: row for row in manifest["runs"]}
        runs = []
        for bound_run in block["runs"]:
            row = rows[bound_run["index"]]
            smoke = launch.parent / "smoke" / output(row=row)
            reference = Path(bound_run["reference"])
            try:
                job = (
                    None
                    if (reference / "host.json").is_file()
                    else reference_job(
                        reference_root=Path(block["reference_root"]),
                        index=bound_run["reference_index"],
                    )
                )
                summary = compare_run(smoke=smoke, reference=reference, reference_job=job)
            except (Divergence, CpuMismatch) as error:
                raise type(error)(
                    f"row {row['index']} ({row['arm']}, seed {row['seed']}) against "
                    f"{bound_run['reference']}: {error}"
                ) from error
            runs.append({
                "index": row["index"],
                "arm": row["arm"],
                "seed": row["seed"],
                "smoke": str(smoke),
                "reference": bound_run["reference"],
                **summary,
            })
        record = {
            "binding": bound,
            "passed": True,
            "reference_root": block["reference_root"],
            "reference_manifest_sha256": block["reference_manifest_sha256"],
            "allowed_argv_differences": block["allowed_argv_differences"],
            "runs": runs,
        }
    with (launch / RECORD).open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--launch", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = check(launch=args.launch.resolve())
    except (Divergence, CpuMismatch) as error:
        sys.exit(f"Equivalence check FAILED: {error}")
    print(json.dumps({key: value for key, value in record.items() if key != "runs"}))


if __name__ == "__main__":
    main()
