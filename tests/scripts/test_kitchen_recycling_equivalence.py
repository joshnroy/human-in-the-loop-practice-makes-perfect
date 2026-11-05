"""The launch-time equivalence guard: a smoke must reproduce a recorded reference run.

Every run here is synthetic and tiny, shaped like kitchen v10's real smoke and full-run
outputs: the smoke stops at its 3,200-step budget mid-skill, the reference practises on.
"""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.kitchen_recycling import equivalence, prepare, worker

CPU = "AMD EPYC 9654 96-Core Processor"


def event(**fields: object) -> dict:
    """A step event as the protocol writes it, wall-clock fields included."""
    return {**fields, "timestamp": "2026-11-04T16:46:32+00:00", "elapsed_seconds": 7.6}


SHARED_EVENTS = [
    event(event="measurement", index=0, practice_steps=0, policy_file_sha256="aa"),
    event(event="session_start", cycle=0, practice_steps=0),
    event(
        event="robot",
        cycle=0,
        label="PickCube(robot, cube_0, cuboid_barrier, bin_0)",
        start_step=0,
        end_step=174,
        interrupted=False,
        interruption=None,
    ),
    event(event="human", cycle=0, start_step=174, end_step=175, skill="ResetCubeBinSameSide"),
]
# The smoke's budget cuts its next skill short and forces a learning update.
SMOKE_TAIL = [
    event(
        event="robot",
        cycle=0,
        label="PickCube(robot, cube_0, cuboid_barrier, bin_0)",
        start_step=175,
        end_step=3200,
        interrupted=True,
        interruption="practice_step_budget",
    ),
    event(event="learning_update", cycle=0, reason="practice_step_budget"),
    event(event="measurement", index=1, practice_steps=3200, policy_file_sha256="bb"),
    event(event="practice_end", practice_steps=3200, reason="practice_step_budget"),
]
REFERENCE_TAIL = [
    event(event="measurement", index=1, practice_steps=3200, policy_file_sha256="cc"),
    event(
        event="robot",
        cycle=0,
        label="PickCube(robot, cube_0, cuboid_barrier, bin_0)",
        start_step=175,
        end_step=3400,
        interrupted=False,
        interruption=None,
    ),
]


def decision(*, number: int, atoms: list[str], address: str) -> dict:
    return {
        "event": "decision",
        "decision": number,
        "atoms": atoms,
        "action": f"<Skill PickCube at {address}>",
        "value": 0.25 * number,
        "search_duration_seconds": 4.2 + number,
        "timestamp": "2026-11-04T16:49:56+00:00",
    }


def decisions(*, atoms: list[str], address: str, tail: list[dict]) -> list[dict]:
    head = []
    for number in range(1, 4):
        head += [
            decision(number=number, atoms=atoms, address=address),
            {"event": "dispatch", "decision": number, "elapsed_seconds": 1.0},
        ]
        if number < 3:
            head.append({"event": "outcome", "decision": number, "success": True})
    return head + tail


SMOKE_DECISIONS = decisions(
    atoms=["HandEmpty(robot)", "OnGround(cube_0)"],
    address="0x7f00aa",
    tail=[
        {"event": "outcome", "decision": 3, "success": False, "cost": 30.25},
        {"event": "refit", "decision": 3, "training_examples": {"PickCube": 3}},
    ],
)
# Same decisions, the frozenset written in another hash order, at another address.
REFERENCE_DECISIONS = decisions(
    atoms=["OnGround(cube_0)", "HandEmpty(robot)"],
    address="0x55deadbeef",
    tail=[{"event": "outcome", "decision": 3, "success": True, "cost": 32.25}],
)


def task(*, number: int, solved: bool = False) -> dict:
    return {
        "task": number,
        "seed": 1000 + number,
        "goal": "InBin(cube_0, bin_0)",
        "solved": solved,
        "steps": 255,
        "reason": "no_plan",
    }


def write_run(
    *,
    directory: Path,
    events: list[dict],
    decision_log: list[dict],
    tasks: int,
    cpu: str | None = CPU,
    decision_file: str = "pomdp_decisions.jsonl",
) -> Path:
    directory.mkdir(parents=True)
    (directory / "step_events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in events)
    )
    (directory / decision_file).write_text(
        "".join(json.dumps(record) + "\n" for record in decision_log)
    )
    checkpoints = [
        {
            "index": index,
            "complete": True,
            "tasks": [task(number=number, solved=index == 1) for number in range(tasks)],
            "num_total": tasks,
            "policy_file_sha256": f"{index}{tasks}",
            "elapsed_seconds": 50.0 + tasks,
        }
        for index in (0, 1)
    ]
    (directory / "step_evaluations.json").write_text(json.dumps(checkpoints))
    for number in range(tasks):
        task_directory = directory / "evaluations" / "0000" / "tasks" / f"{number:02d}"
        task_directory.mkdir(parents=True)
        (task_directory / "results.json").write_text(
            json.dumps({"tasks": [task(number=number)], "elapsed_seconds": 3.0 + tasks})
        )
        (task_directory / "actions.jsonl").write_text(
            json.dumps({"task": number, "label": "PickCube", "timestamp": f"t{tasks}"}) + "\n"
        )
    if cpu is not None:
        node = f"della-h1{tasks}n1"
        (directory / "host.json").write_text(
            json.dumps({"cpu_model": cpu, "hostname": node, "slurm": {"SLURMD_NODENAME": node}})
        )
    return directory


def pair(
    *,
    root: Path,
    reference_events: list[dict] | None = None,
    reference_decisions: list[dict] | None = None,
    reference_cpu: str | None = CPU,
) -> tuple[Path, Path]:
    smoke = write_run(
        directory=root / "smoke",
        events=SHARED_EVENTS + SMOKE_TAIL,
        decision_log=SMOKE_DECISIONS,
        tasks=2,
    )
    reference = write_run(
        directory=root / "reference",
        events=reference_events or SHARED_EVENTS + REFERENCE_TAIL,
        decision_log=reference_decisions or REFERENCE_DECISIONS,
        tasks=4,
        cpu=reference_cpu,
    )
    return smoke, reference


def test_a_matching_pair_passes(*, tmp_path: Path) -> None:
    """Timing, file hashes, set order and addresses differ; the post-budget tail differs;
    every decision the smoke made before its budget matches."""
    smoke, reference = pair(root=tmp_path)
    summary = equivalence.compare_run(smoke=smoke, reference=reference)
    assert summary["step_events"] == 4
    assert summary["decisions"] == {"file": "pomdp_decisions.jsonl", "records": 8}
    assert summary["evaluation_tasks"] == 2
    assert summary["cpu_identity"] == {"basis": "host.json cpu_model", "cpu_model": CPU}


def test_a_one_event_divergence_fails_naming_the_event(*, tmp_path: Path) -> None:
    events = copy.deepcopy(SHARED_EVENTS + REFERENCE_TAIL)
    events[3]["skill"] = "ResetCubeBinOppositeSide"
    smoke, reference = pair(root=tmp_path, reference_events=events)
    with pytest.raises(equivalence.Divergence) as raised:
        equivalence.compare_run(smoke=smoke, reference=reference)
    message = str(raised.value)
    assert "step_events.jsonl" in message
    assert "record 3" in message
    assert "ResetCubeBinOppositeSide" in message
    assert "ResetCubeBinSameSide" in message


def test_a_decision_divergence_fails_naming_the_record(*, tmp_path: Path) -> None:
    log = copy.deepcopy(REFERENCE_DECISIONS)
    log[3]["value"] = 0.75
    smoke, reference = pair(root=tmp_path, reference_decisions=log)
    with pytest.raises(equivalence.Divergence, match=r"pomdp_decisions\.jsonl.*record 3"):
        equivalence.compare_run(smoke=smoke, reference=reference)


def test_a_reference_that_stops_early_diverges(*, tmp_path: Path) -> None:
    smoke, reference = pair(root=tmp_path, reference_decisions=REFERENCE_DECISIONS[:4])
    with pytest.raises(equivalence.Divergence, match="reference ends"):
        equivalence.compare_run(smoke=smoke, reference=reference)


def test_an_evaluation_divergence_fails_naming_the_task(*, tmp_path: Path) -> None:
    smoke, reference = pair(root=tmp_path)
    checkpoints = json.loads((reference / "step_evaluations.json").read_text())
    checkpoints[0]["tasks"][1]["steps"] = 500
    (reference / "step_evaluations.json").write_text(json.dumps(checkpoints))
    with pytest.raises(equivalence.Divergence, match=r"checkpoint 0 task 1"):
        equivalence.compare_run(smoke=smoke, reference=reference)


def test_a_cpu_model_mismatch_refuses(*, tmp_path: Path) -> None:
    smoke, reference = pair(root=tmp_path, reference_cpu="Intel(R) Xeon(R) Gold 6248R")
    with pytest.raises(equivalence.CpuMismatch, match="Xeon.*EPYC|EPYC.*Xeon"):
        equivalence.compare_run(smoke=smoke, reference=reference)


def test_a_smoke_without_recorded_host_refuses(*, tmp_path: Path) -> None:
    smoke, reference = pair(root=tmp_path)
    (smoke / "host.json").unlink()
    with pytest.raises(equivalence.CpuMismatch, match="host.json"):
        equivalence.compare_run(smoke=smoke, reference=reference, reference_job="16840531_1")


def test_a_legacy_reference_without_a_slurm_job_refuses(*, tmp_path: Path) -> None:
    smoke, reference = pair(root=tmp_path, reference_cpu=None)
    with pytest.raises(equivalence.CpuMismatch, match="host.json.*Slurm job"):
        equivalence.compare_run(smoke=smoke, reference=reference)


GENOA = {"amd", "genoa", "rh9", "nvme"}
CASCADE = {"intel", "cascade", "rh9"}


def legacy_slurm(*, features: dict[str, set[str]]):
    """Stand-ins for sacct (the legacy reference job's node) and sinfo (a node's
    features), so no test needs a cluster."""
    return (
        patch.object(equivalence, "slurm_node", return_value="della-h16n13"),
        patch.object(equivalence, "node_features", side_effect=lambda *, node: features[node]),
    )


def test_a_legacy_reference_is_accepted_on_a_shared_genoa_feature(*, tmp_path: Path) -> None:
    """Kitchen v10 wrote no host.json: its CPU identity is its Slurm node's feature."""
    smoke, reference = pair(root=tmp_path, reference_cpu=None)
    node, features = legacy_slurm(features={"della-h12n1": GENOA, "della-h16n13": GENOA})
    with node as sacct, features:
        summary = equivalence.compare_run(
            smoke=smoke, reference=reference, reference_job="16840531_1"
        )
    sacct.assert_called_once_with(job="16840531_1")
    assert summary["cpu_identity"] == {
        "basis": "slurm-feature:genoa",
        "smoke_node": "della-h12n1",
        "reference_node": "della-h16n13",
        "reference_job": "16840531_1",
    }


@pytest.mark.parametrize(
    ("smoke_features", "reference_features"), [(GENOA, CASCADE), (CASCADE, GENOA)]
)
def test_a_legacy_reference_is_refused_without_a_shared_genoa_feature(
    *, tmp_path: Path, smoke_features: set[str], reference_features: set[str]
) -> None:
    smoke, reference = pair(root=tmp_path, reference_cpu=None)
    node, features = legacy_slurm(
        features={"della-h12n1": smoke_features, "della-h16n13": reference_features}
    )
    with node, features, pytest.raises(equivalence.CpuMismatch, match="genoa"):
        equivalence.compare_run(smoke=smoke, reference=reference, reference_job="16840531_1")


def test_a_reference_with_host_json_still_needs_the_exact_model(*, tmp_path: Path) -> None:
    """The Slurm feature is only for references that recorded no model."""
    smoke, reference = pair(root=tmp_path, reference_cpu="AMD EPYC 9655 96-Core Processor")
    node, features = legacy_slurm(features={"della-h12n1": GENOA, "della-h16n13": GENOA})
    with node, features, pytest.raises(equivalence.CpuMismatch, match="9655"):
        equivalence.compare_run(smoke=smoke, reference=reference, reference_job="16840531_1")


@pytest.mark.parametrize(
    ("output", "node"),
    [
        ("16840531_1|della-h16n13\n", "della-h16n13"),
        ("16840531_1|della-h16n13\n16840531_1|della-h16n14\n", None),
        ("16840531_1|della-h16n[13-14]\n", None),
        ("", None),
    ],
)
def test_sacct_must_name_exactly_one_node(*, output: str, node: str | None) -> None:
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=output)
    with patch.object(equivalence.subprocess, "run", return_value=completed) as run:
        if node is None:
            with pytest.raises(equivalence.CpuMismatch, match="sacct"):
                equivalence.slurm_node(job="16840531_1")
        else:
            assert equivalence.slurm_node(job="16840531_1") == node
    assert run.call_args.args[0][0] == "sacct"


def test_node_features_are_read_from_sinfo() -> None:
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="amd,genoa,rh9,nvme\n")
    with patch.object(equivalence.subprocess, "run", return_value=completed) as run:
        assert equivalence.node_features(node="della-h16n13") == GENOA
    assert run.call_args.args[0][0] == "sinfo"


@pytest.mark.parametrize("logs", [[], ["a-full-16840531_1.out", "a-full-16850000_1.out"]])
def test_a_legacy_reference_job_must_be_one_array_task(*, tmp_path: Path, logs: list) -> None:
    (tmp_path / "logs").mkdir()
    for name in [*logs, "a-full-16840531_11.out", "a-smoke-16840529_4294967294.out"]:
        (tmp_path / "logs" / name).write_text("")
    with pytest.raises(equivalence.CpuMismatch, match="logs"):
        equivalence.reference_job(reference_root=tmp_path, index=1)


def test_a_legacy_reference_job_is_read_from_its_full_array_log(*, tmp_path: Path) -> None:
    (tmp_path / "logs").mkdir()
    for name in ("v10-full-16840531_1.out", "v10-full-16840531_11.out", "gate-16840530.out"):
        (tmp_path / "logs" / name).write_text("")
    assert equivalence.reference_job(reference_root=tmp_path, index=1) == "16840531_1"


def test_a_smoke_with_nothing_before_its_budget_refuses(*, tmp_path: Path) -> None:
    smoke = write_run(
        directory=tmp_path / "smoke", events=SMOKE_TAIL, decision_log=SMOKE_DECISIONS, tasks=2
    )
    reference = write_run(
        directory=tmp_path / "reference",
        events=SHARED_EVENTS,
        decision_log=REFERENCE_DECISIONS,
        tasks=4,
    )
    with pytest.raises(equivalence.Divergence, match="nothing to compare"):
        equivalence.compare_run(smoke=smoke, reference=reference)


# ------------------------------------------------- #681's decision-log schema change
# #681 removed Skill.practice_cost and the outcome records' cost-observation fields after
# kitchen v10 ran. A v10 reference logs them; a smoke from today's source cannot. Exactly
# those three fields are ignored, and nothing else.

SKILL_REPR_V10 = (
    "skill=Skill(name='PickCube', parameters=(), preconditions=frozenset(), "
    "param_dim=0, practice_cost=1.0) objects=(Object(name='robot'),)"
)
SKILL_REPR_NOW = (
    "skill=Skill(name='PickCube', parameters=(), preconditions=frozenset(), "
    "param_dim=0) objects=(Object(name='robot'),)"
)


def skill_dict(**extra: object) -> dict:
    return {"name": "PickCube", "parameters": [], "preconditions": [], "param_dim": 0, **extra}


def outcome_v10(**changes: object) -> dict:
    return {
        "event": "outcome",
        "decision": 3,
        "success": True,
        "action": {"skill": skill_dict(practice_cost=1.0), "objects": ["robot"]},
        "configured_cost_observation": None,
        "cost_observation_source": "host_control_steps",
        "estimated_costs": {SKILL_REPR_V10: 2.09},
        **changes,
    }


def outcome_now(**changes: object) -> dict:
    return {
        "event": "outcome",
        "decision": 3,
        "success": True,
        "action": {"skill": skill_dict(), "objects": ["robot"]},
        "estimated_costs": {SKILL_REPR_NOW: 2.09},
        **changes,
    }


def test_the_681_schema_fields_are_ignored() -> None:
    equivalence.compare_records(
        name="pomdp_decisions.jsonl", smoke=[outcome_now()], reference=[outcome_v10()]
    )
    equivalence.compare_records(
        name="pomdp_decisions.jsonl",
        smoke=[{"event": "decision", "chosen": SKILL_REPR_NOW}],
        reference=[{"event": "decision", "chosen": SKILL_REPR_V10}],
    )


@pytest.mark.parametrize(
    ("smoke", "reference"),
    [
        # A decision that differs is still a divergence.
        (outcome_now(success=False), outcome_v10()),
        # So is a different cost estimate behind the same renamed key.
        (outcome_now(estimated_costs={SKILL_REPR_NOW: 2.10}), outcome_v10()),
        # practice_cost is ignored only on a serialised Skill, not on any dict.
        (
            {"event": "execution_cost", "cost": {"value": 1}},
            {"event": "execution_cost", "cost": {"value": 1, "practice_cost": 1.0}},
        ),
        # The cost-observation fields are ignored only on outcome records.
        (
            {"event": "execution_cost", "cost": 1},
            {"event": "execution_cost", "cost": 1, "cost_observation_source": "host"},
        ),
        # Only a numeric practice_cost at the end of a Skill repr is stripped.
        (
            {"event": "decision", "chosen": SKILL_REPR_NOW},
            {"event": "decision", "chosen": SKILL_REPR_NOW.replace("PickCube", "Pick")},
        ),
    ],
)
def test_anything_else_still_diverges(*, smoke: dict, reference: dict) -> None:
    with pytest.raises(equivalence.Divergence):
        equivalence.compare_records(
            name="pomdp_decisions.jsonl", smoke=[smoke], reference=[reference]
        )


# ---------------------------------------------------------------- prepare.py binding


def run_prepare(*, out: Path, tag: str, extra: tuple[str, ...] = ()) -> subprocess.Popen:
    return subprocess.run(
        [
            sys.executable,
            str(Path(prepare.__file__)),
            *("--out", str(out), "--tag", tag),
            *("--practice-step-budget", "65000", "--evaluation-workers", "4"),
            *("--human-reset-costs", "5", "100", "--max-parallel", "18"),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


WAIVER = ("--equivalence-reference", "none", "--equivalence-waiver", "unit-test package")


@pytest.fixture
def reference_launch(*, tmp_path: Path) -> Path:
    """A previous launch's package, its manifest where Della keeps it: <root>/launch."""
    out = tmp_path / "kitchen-recycling-reference" / "launch"
    assert run_prepare(out=out, tag="kitchen-recycling-reference", extra=WAIVER).returncode == 0
    return out.parent


def test_prepare_binds_each_smoke_row_to_its_reference_run(
    *, tmp_path: Path, reference_launch: Path
) -> None:
    waived = tmp_path / "waived"
    bound = tmp_path / "bound"
    assert run_prepare(out=waived, tag="kitchen-recycling-new", extra=WAIVER).returncode == 0
    result = run_prepare(
        out=bound,
        tag="kitchen-recycling-new",
        extra=("--equivalence-reference", str(reference_launch)),
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((bound / "manifest.json").read_text())
    block = manifest.pop("equivalence")
    # Only the manifest's equivalence key differs; rows, commands and scripts do not.
    other = json.loads((waived / "manifest.json").read_text())
    assert other.pop("equivalence") == {"reference_root": None, "waiver": "unit-test package"}
    assert manifest == other
    for name in ("commands.txt", "smoke.sbatch", "full.sbatch"):
        assert (bound / name).read_bytes() == (waived / name).read_bytes()
    reference_manifest = reference_launch / "launch" / "manifest.json"
    assert block == {
        "reference_root": str(reference_launch),
        "reference_manifest_sha256": hashlib.sha256(reference_manifest.read_bytes()).hexdigest(),
        "allowed_argv_differences": [],
        "runs": [
            {
                "index": index,
                "reference_index": index,
                "reference": f"{reference_launch}/results/reset-cost-5/{arm}/0",
            }
            for index, arm in enumerate(prepare.ARMS)
        ],
    }


def test_prepare_refuses_an_undeclared_argv_difference(
    *, tmp_path: Path, reference_launch: Path
) -> None:
    result = run_prepare(
        out=tmp_path / "new",
        tag="kitchen-recycling-new",
        extra=(
            *("--repeated-stop-limit", "100"),
            *("--equivalence-reference", str(reference_launch)),
        ),
    )
    assert result.returncode != 0
    assert "--repeated-stop-limit" in result.stderr
    assert not (tmp_path / "new").exists()


def test_prepare_accepts_a_declared_argv_difference(
    *, tmp_path: Path, reference_launch: Path
) -> None:
    result = run_prepare(
        out=tmp_path / "new",
        tag="kitchen-recycling-new",
        extra=(
            *("--repeated-stop-limit", "100"),
            *("--equivalence-reference", str(reference_launch)),
            *("--equivalence-allow-difference", "repeated-stop-limit"),
        ),
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "new" / "manifest.json").read_text())
    assert manifest["equivalence"]["allowed_argv_differences"] == ["--repeated-stop-limit"]


def test_prepare_reads_a_local_copy_of_a_remote_reference_manifest(
    *, tmp_path: Path, reference_launch: Path
) -> None:
    """The package is prepared on agni; the reference lives on Della's scratch."""
    remote = "/scratch/gpfs/TSILVER/jr2860/kitchen-recycling/kitchen-recycling-reference"
    result = run_prepare(
        out=tmp_path / "new",
        tag="kitchen-recycling-new",
        extra=(
            *("--equivalence-reference", remote),
            *(
                "--equivalence-reference-manifest",
                str(reference_launch / "launch" / "manifest.json"),
            ),
        ),
    )
    assert result.returncode == 0, result.stderr
    block = json.loads((tmp_path / "new" / "manifest.json").read_text())["equivalence"]
    assert block["reference_root"] == remote
    assert block["runs"][1]["reference"] == f"{remote}/results/reset-cost-5/expectimax/0"


def smoke_rows(*, argv: list[str]) -> list[dict]:
    return [
        {"index": index, "arm": arm, "seed": 0, "argv": argv, "costs_sha256": "c" * 64}
        for index, arm in enumerate(prepare.ARMS)
    ]


def test_binding_reads_flags_in_any_order_but_not_any_value() -> None:
    """Kitchen v10 wrote expectimax's --pomdp-search-depth before the engine flags; main
    writes it last. argparse reads both the same; a changed value is still refused."""
    ours = ["--seed", "0", "--practice-cost-config", "/v11/costs.json", "--depth", "6", "--x"]
    theirs = ["--seed", "0", "--depth", "6", "--x", "--practice-cost-config", "/v10/costs.json"]
    bound = prepare.bind_equivalence(
        rows=smoke_rows(argv=ours),
        reference={"runs": smoke_rows(argv=theirs)},
        reference_root="/v10",
        allowed=[],
    )
    assert [run["reference"] for run in bound] == [f"/v10/results/{arm}/0" for arm in prepare.ARMS]
    for changed in (
        ["--seed", "0", "--depth", "7", "--x", "--practice-cost-config", "/v10/costs.json"],
        ["--seed", "0", "--depth", "6", "--practice-cost-config", "/v10/costs.json"],
        [*theirs, "--depth", "6"],
    ):
        with pytest.raises(ValueError, match="differs from reference"):
            prepare.bind_equivalence(
                rows=smoke_rows(argv=ours),
                reference={"runs": smoke_rows(argv=changed)},
                reference_root="/v10",
                allowed=[],
            )


def test_a_waiver_is_recorded_in_the_manifest(*, tmp_path: Path) -> None:
    result = run_prepare(
        out=tmp_path / "new",
        tag="kitchen-recycling-new",
        extra=(
            *("--equivalence-reference", "none"),
            *("--equivalence-waiver", "v12 adds --repeated-stop-limit; meant to differ"),
        ),
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "new" / "manifest.json").read_text())
    assert manifest["equivalence"] == {
        "reference_root": None,
        "waiver": "v12 adds --repeated-stop-limit; meant to differ",
    }


@pytest.mark.parametrize(
    "extra",
    [
        (),  # Mandatory from v11: a package binds a reference or records a waiver.
        ("--equivalence-reference", "none"),
        ("--equivalence-reference", "none", "--equivalence-waiver", "  "),
        ("--equivalence-waiver", "no reference named"),
        ("--equivalence-allow-difference", "repeated-stop-limit"),
    ],
)
def test_prepare_refuses_an_unrecorded_or_half_stated_opt_out(
    *, tmp_path: Path, extra: tuple[str, ...]
) -> None:
    result = run_prepare(out=tmp_path / "new", tag="kitchen-recycling-new", extra=extra)
    assert result.returncode != 0
    assert not (tmp_path / "new").exists()


def test_a_waiver_cannot_accompany_a_reference(*, tmp_path: Path, reference_launch: Path) -> None:
    result = run_prepare(
        out=tmp_path / "new",
        tag="kitchen-recycling-new",
        extra=(
            *("--equivalence-reference", str(reference_launch)),
            *("--equivalence-waiver", "both"),
        ),
    )
    assert result.returncode != 0


DP = (
    *("--arms", "ees", "expectimax", "curtis", "dp"),
    *("--dp-episode-budget", "188", "--dp-repeated-stop-limit", "100"),
)


def test_prepare_waives_a_new_arm_and_binds_the_others(
    *, tmp_path: Path, reference_launch: Path
) -> None:
    """A new arm has no reference run; the arms the reference ran are still checked."""
    result = run_prepare(
        out=tmp_path / "new",
        tag="kitchen-recycling-new",
        extra=(
            *DP,
            *("--equivalence-reference", str(reference_launch)),
            *("--equivalence-waive-arm", "dp"),
            *("--equivalence-waiver", "new DP arm, no prior reference"),
        ),
    )
    assert result.returncode == 0, result.stderr
    block = json.loads((tmp_path / "new" / "manifest.json").read_text())["equivalence"]
    assert block["waived_arms"] == ["dp"]
    assert block["waiver"] == "new DP arm, no prior reference"
    assert block["runs"] == [
        {
            "index": index,
            "reference_index": index,
            "reference": f"{reference_launch}/results/reset-cost-5/{arm}/0",
        }
        for index, arm in enumerate(prepare.ARMS)
    ]


@pytest.mark.parametrize(
    "extra",
    [
        (),  # The reference has no DP run to bind.
        ("--equivalence-waive-arm", "dp"),  # A waived arm needs a stated reason.
        ("--equivalence-waiver", "why"),  # A reason needs the arm it waives.
        ("--equivalence-waive-arm", "sac", "--equivalence-waiver", "why"),  # No such arm.
    ],
)
def test_prepare_refuses_a_half_stated_arm_waiver(
    *, tmp_path: Path, reference_launch: Path, extra: tuple[str, ...]
) -> None:
    result = run_prepare(
        out=tmp_path / "new",
        tag="kitchen-recycling-new",
        extra=(*DP, *("--equivalence-reference", str(reference_launch)), *extra),
    )
    assert result.returncode != 0
    assert not (tmp_path / "new").exists()


# ---------------------------------------------------------------- the gate's check


def bound_launch(*, tmp_path: Path, block: dict) -> Path:
    launch = tmp_path / "deploy" / "launch"
    launch.mkdir(parents=True)
    rows = [
        {"index": index, "arm": arm, "seed": 0, "argv": [], "output": f"reset-cost-5/{arm}/0"}
        for index, arm in enumerate(prepare.ARMS)
    ]
    (launch / "manifest.json").write_text(json.dumps({"runs": rows, "equivalence": block}))
    for name in ("snapshot.json", "runtime-lock.json"):
        (launch / name).write_text("{}")
    return launch


def bound_pair(
    *,
    tmp_path: Path,
    reference_decisions: list[dict] | None = None,
    legacy_curtis: bool = False,
) -> Path:
    reference_root = tmp_path / "reference-root"
    (reference_root / "launch").mkdir(parents=True)
    (reference_root / "launch" / "manifest.json").write_text('{"runs": []}')
    (reference_root / "logs").mkdir()
    (reference_root / "logs" / "ref-full-16840531_2.out").write_text("")
    runs = []
    for index, arm in enumerate(prepare.ARMS):
        smoke, reference = pair(
            root=tmp_path / arm,
            reference_decisions=reference_decisions if arm == "curtis" else None,
            reference_cpu=None if legacy_curtis and arm == "curtis" else CPU,
        )
        target = tmp_path / "deploy" / "smoke" / "reset-cost-5" / arm / "0"
        target.parent.mkdir(parents=True, exist_ok=True)
        smoke.rename(target)
        runs.append({"index": index, "reference_index": index, "reference": str(reference)})
    return bound_launch(
        tmp_path=tmp_path,
        block={
            "reference_root": str(reference_root),
            "reference_manifest_sha256": hashlib.sha256(b'{"runs": []}').hexdigest(),
            "allowed_argv_differences": [],
            "runs": runs,
        },
    )


def test_the_gate_check_records_a_pass_bound_to_the_launch(*, tmp_path: Path) -> None:
    launch = bound_pair(tmp_path=tmp_path)
    record = equivalence.check(launch=launch)
    assert record == json.loads((launch / equivalence.RECORD).read_text())
    assert record["passed"] is True
    assert record["binding"] == worker.binding(launch=launch)
    assert [run["arm"] for run in record["runs"]] == list(prepare.ARMS)
    assert all(run["step_events"] == 4 for run in record["runs"])


def test_the_gate_check_records_which_cpu_identity_it_used(*, tmp_path: Path) -> None:
    """A legacy reference row's job comes from its full-array log; the record says the
    Slurm feature, not a CPU model, matched it."""
    launch = bound_pair(tmp_path=tmp_path, legacy_curtis=True)
    node, features = legacy_slurm(features={"della-h12n1": GENOA, "della-h16n13": GENOA})
    with node as sacct, features:
        record = equivalence.check(launch=launch)
    sacct.assert_called_once_with(job="16840531_2")
    identities = [run["cpu_identity"] for run in record["runs"]]
    assert identities[:2] == [{"basis": "host.json cpu_model", "cpu_model": CPU}] * 2
    assert identities[2] == {
        "basis": "slurm-feature:genoa",
        "smoke_node": "della-h12n1",
        "reference_node": "della-h16n13",
        "reference_job": "16840531_2",
    }


def test_the_gate_check_fails_without_writing_a_record(*, tmp_path: Path) -> None:
    log = copy.deepcopy(REFERENCE_DECISIONS)
    log[0]["value"] = 9.0
    launch = bound_pair(tmp_path=tmp_path, reference_decisions=log)
    with pytest.raises(equivalence.Divergence, match=r"curtis.*pomdp_decisions\.jsonl"):
        equivalence.check(launch=launch)
    assert not (launch / equivalence.RECORD).exists()


def test_the_gate_check_refuses_a_reference_manifest_that_changed(*, tmp_path: Path) -> None:
    launch = bound_pair(tmp_path=tmp_path)
    (tmp_path / "reference-root" / "launch" / "manifest.json").write_text('{"runs": [1]}')
    with pytest.raises(ValueError, match="reference manifest"):
        equivalence.check(launch=launch)


def test_the_gate_check_records_a_waiver(*, tmp_path: Path) -> None:
    launch = bound_launch(
        tmp_path=tmp_path, block={"reference_root": None, "waiver": "meant to differ"}
    )
    record = equivalence.check(launch=launch)
    assert record == {
        "binding": worker.binding(launch=launch),
        "passed": False,
        "waived": True,
        "waiver": "meant to differ",
    }
    assert json.loads((launch / equivalence.RECORD).read_text()) == record


def test_the_gate_check_carries_an_arm_waiver_into_its_record(*, tmp_path: Path) -> None:
    launch = bound_pair(tmp_path=tmp_path)
    manifest = json.loads((launch / "manifest.json").read_text())
    manifest["equivalence"] |= {"waived_arms": ["dp"], "waiver": "new DP arm"}
    (launch / "manifest.json").write_text(json.dumps(manifest))
    record = equivalence.check(launch=launch)
    assert record["passed"] is True
    assert (record["waived_arms"], record["waiver"]) == (["dp"], "new DP arm")
    assert [run["arm"] for run in record["runs"]] == list(prepare.ARMS)


# ---------------------------------------------------------------- worker.py


def test_the_full_worker_requires_the_record_in_authorization_provenance(*, tmp_path: Path) -> None:
    launch = bound_launch(
        tmp_path=tmp_path, block={"reference_root": None, "waiver": "meant to differ"}
    )
    manifest = json.loads((launch / "manifest.json").read_text())
    with pytest.raises(FileNotFoundError):
        worker.check_equivalence(launch=launch, manifest=manifest)
    record = equivalence.check(launch=launch)
    with pytest.raises(FileNotFoundError):
        worker.check_equivalence(launch=launch, manifest=manifest)
    provenance = launch / "authorization-provenance.json"
    provenance.write_text(json.dumps({"phase": "full", "equivalence": {"waived": True}}))
    with pytest.raises(ValueError, match="provenance"):
        worker.check_equivalence(launch=launch, manifest=manifest)
    provenance.write_text(json.dumps({"phase": "full", "equivalence": record}))
    worker.check_equivalence(launch=launch, manifest=manifest)
    # Mandatory from v11: a manifest with neither a reference nor a waiver is refused.
    with pytest.raises(ValueError, match="neither an equivalence reference nor a waiver"):
        worker.check_equivalence(launch=launch, manifest={"runs": []})


def test_the_full_worker_refuses_a_failed_or_stale_record(*, tmp_path: Path) -> None:
    launch = bound_pair(tmp_path=tmp_path)
    manifest = json.loads((launch / "manifest.json").read_text())
    record = equivalence.check(launch=launch)
    (launch / "authorization-provenance.json").write_text(json.dumps({"equivalence": record}))
    worker.check_equivalence(launch=launch, manifest=manifest)
    (launch / "snapshot.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="binding"):
        worker.check_equivalence(launch=launch, manifest=manifest)


def test_a_run_records_its_host(*, tmp_path: Path) -> None:
    """Behaviour-neutral: written by the worker after the CLI has exited."""
    launch = tmp_path / "launch"
    launch.mkdir()
    for name in ("manifest.json", "snapshot.json", "runtime-lock.json"):
        (launch / name).write_text("{}")

    def completed(*args: list[str], **_: object) -> None:
        (command,) = args  # subprocess.run's own signature
        output = Path(command[command.index("--output-dir") + 1])
        assert not (output / "host.json").exists()
        (output / "status.json").write_text('{"phase": "complete", "practice_steps": 3200}')
        (output / "step_evaluations.json").write_text(
            '[{"complete": true, "practice_steps": 3200}]'
        )
        (output / "step_events.jsonl").write_text(
            '{"event": "practice_end", "reason": "practice_step_budget"}\n'
        )

    row = {"arm": "ees", "seed": 0, "argv": []}
    with (
        patch.object(worker.subprocess, "run", side_effect=completed),
        patch.dict(worker.os.environ, {"SLURM_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "4"}),
    ):
        worker.run(launch=launch, row=row, smoke=True, manifest={})
    host = json.loads((tmp_path / "smoke" / "ees" / "0" / "host.json").read_text())
    assert host["cpu_model"] == worker.cpu_model()
    assert host["cpu_model"]
    assert host["slurm"]["SLURM_JOB_ID"] == "123"
    assert host["slurm"]["SLURM_ARRAY_TASK_ID"] == "4"
