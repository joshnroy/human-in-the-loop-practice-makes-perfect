"""Offline packaging safety checks; no simulator, SSH or scheduler calls."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kitchen_prepare", HERE / "prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
worker_spec = importlib.util.spec_from_file_location("kitchen_worker", HERE / "worker.py")
worker = importlib.util.module_from_spec(worker_spec)
worker_spec.loader.exec_module(worker)

# prepare.py's default package, tag kitchen-recycling-unit-test, as it was before the
# budget and evaluation-worker options existed (the package v5-v8 launched from).
V8_STYLE_DIGESTS = {
    "commands.txt": "d4ce2a3de8ea7aca9e1996c66a507a69b7f9e93ec0ff500bc3ea9eef811d0aa6",
    "full.sbatch": "f2aaa6420590d9e6d93f148e7da1e76cbeb5d63c92e19601690b89cfa437562d",
    "manifest.json": "6ad903d6c2681851141b45ab998b50e6b1bfc7abab2b3dc10c900d0575256fce",
    "smoke.sbatch": "adeed680afc6462eb1f0a4ee8b35380dd49bd7f180ffe506876d4b5e4e72e5f8",
}
ITERATION = {"practice_step_budget": 65000, "evaluation_workers": 4}
# The same tag's v9-style package (65k, four evaluation workers, 6 CPUs / 24G) as main
# wrote it before the reset-cost sweep and --constraint existed.
V9_STYLE_ARGS = (
    *("--practice-step-budget", "65000", "--evaluation-workers", "4"),
    *("--cpus-per-task", "6", "--memory", "24G"),
)
V9_STYLE_DIGESTS = {
    "commands.txt": "258ab4d84d7b77297f49957de6ab7435c64f4b6ca4a79e2737e9058184b08881",
    "full.sbatch": "781eceae5ad2847cd37c1b1648aea918ded6ea23b468adc182cced61192b581c",
    "manifest.json": "65a262430617b9dd1eb4de0b4453ff32183305691f0be98e26dcc9902d7e31f5",
    "smoke.sbatch": "81506e46504fde7f8872df149f922739d5e208c97488ab49b83c66372f498d4f",
}


def prepared(*, directory: str, extra: tuple[str, ...] = ()) -> Path:
    output = Path(directory) / "launch"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "prepare.py"),
            "--out",
            str(output),
            "--tag",
            "kitchen-recycling-unit-test",
            *extra,
        ],
        check=True,
        capture_output=True,
        cwd=directory,  # Not the repository: prepare.py must not depend on where it runs.
    )
    return output


def completed_run(*, practice_steps: int, endpoint: str = "practice_step_budget"):
    """A subprocess.run stand-in that leaves a completed run's files behind."""

    def run(*args, **kwargs):
        (command,) = args
        output = Path(command[command.index("--output-dir") + 1])
        (output / "status.json").write_text(
            json.dumps({"phase": "complete", "practice_steps": practice_steps})
        )
        (output / "step_evaluations.json").write_text(
            json.dumps([{"complete": True, "practice_steps": practice_steps}])
        )
        (output / "step_events.jsonl").write_text(
            json.dumps({"event": "practice_end", "reason": endpoint}) + "\n"
        )

    return run


class PackagingTests(unittest.TestCase):
    def test_smoke_timeout_does_not_cap_full_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            launch = Path(directory) / "launch"
            row = {"arm": "ees", "seed": 0, "argv": []}
            for smoke, timeout in ((True, 1800), (False, None)):
                with self.subTest(smoke=smoke):
                    with (
                        patch.object(
                            worker.subprocess, "run", side_effect=RuntimeError("intercepted")
                        ) as run,
                        self.assertRaisesRegex(RuntimeError, "intercepted"),
                    ):
                        worker.run(launch=launch, row=row, smoke=smoke, manifest={})
                    self.assertEqual(run.call_args.kwargs["timeout"], timeout)

    def test_exact_methods_and_budget(self):
        for arm in prepare.ARMS:
            argv = prepare.argv(
                arm=arm, seed=1, cost_file="costs.json", kitchen_config="scene.json"
            )
            self.assertEqual(argv[argv.index("--env") + 1], "kitchen_recycling3d")
            self.assertEqual(argv[argv.index("--practice-step-budget") + 1], "160000")
            self.assertNotIn("--resume", argv)
            self.assertNotIn("--re-run", argv)
            self.assertNotIn("--preset", argv)
            self.assertIn("--no-scene-bg", argv)
            self.assertIn("--no-early-stop", argv)
            self.assertEqual(argv[argv.index("--goal-pursuit-horizon") + 1], "0")
            if arm == "ees":
                self.assertEqual(argv[argv.index("--practice-groundings") + 1], "all")
                self.assertIn("--ees-score-human-skills", argv)
            if arm == "curtis":
                self.assertEqual(argv[argv.index("--pomdp-determinized-variant") + 1], "curtis")
                self.assertEqual(argv[argv.index("--pomdp-max-search-iterations") + 1], "100")

    def test_proposal_is_seed_major_non_submitting_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "launch"
            command = [
                sys.executable,
                str(HERE / "prepare.py"),
                "--out",
                str(output),
                "--tag",
                "kitchen-recycling-unit-test",
                "--max-parallel",
                "4",
            ]
            subprocess.run(command, check=True, capture_output=True)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(
                [(row["seed"], row["arm"]) for row in manifest["runs"]],
                [(seed, arm) for seed in range(3) for arm in prepare.ARMS],
            )
            self.assertNotIn("cost_budget_flag", manifest)
            self.assertEqual(
                manifest["budget_policy"], "160k counted steps; cost logged, not a hard cap"
            )
            self.assertIn("--array=0-8%4", (output / "full.sbatch").read_text())
            self.assertIn("--time=02:00:00", (output / "smoke.sbatch").read_text())
            self.assertIn("--time=48:00:00", (output / "full.sbatch").read_text())
            self.assertNotIn("dependency", (output / "full.sbatch").read_text())
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertFalse((output / "authorization.json").exists())

    def test_defaults_leave_a_v8_style_package_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            output = prepared(directory=directory)
            for name, digest in V8_STYLE_DIGESTS.items():
                with self.subTest(file=name):
                    self.assertEqual(
                        hashlib.sha256((output / name).read_bytes()).hexdigest(), digest
                    )

    def test_no_sweep_leaves_a_v9_style_package_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            output = prepared(directory=directory, extra=V9_STYLE_ARGS)
            self.assertFalse((output / "costs").exists())
            for name, digest in V9_STYLE_DIGESTS.items():
                with self.subTest(file=name):
                    self.assertEqual(
                        hashlib.sha256((output / name).read_bytes()).hexdigest(), digest
                    )

    def test_budget_and_evaluation_workers_reach_every_row(self):
        with tempfile.TemporaryDirectory() as directory:
            output = prepared(
                directory=directory,
                extra=(
                    "--practice-step-budget",
                    "65000",
                    "--evaluation-workers",
                    "4",
                    "--cpus-per-task",
                    "6",
                    "--memory",
                    "24G",
                ),
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["practice_step_budget"], 65000)
            self.assertEqual(manifest["evaluation_workers"], 4)
            self.assertEqual(
                manifest["budget_policy"], "65k counted steps; cost logged, not a hard cap"
            )
            self.assertEqual(manifest["resources"]["cpus"], 6)
            self.assertEqual(manifest["resources"]["memory"], "24G")
            self.assertEqual(worker.contract(manifest=manifest), (65000, 4))
            for row in manifest["runs"]:
                argv = row["argv"]
                self.assertEqual(argv[argv.index("--practice-step-budget") + 1], "65000")
                self.assertEqual(argv[argv.index("--evaluation-workers") + 1], "4")
                self.assertEqual(argv[argv.index("--evaluation-task-rng") + 1], "per-task")
                self.assertEqual(argv[argv.index("--measurement-interval-steps") + 1], "3200")
                self.assertEqual(argv[argv.index("--num-test-tasks") + 1], "10")
                self.assertNotIn("--ees-preview-test-goals", argv)
            for phase in ("smoke", "full"):
                script = (output / f"{phase}.sbatch").read_text()
                self.assertIn("--cpus-per-task=6\n", script)
                self.assertIn("--mem=24G\n", script)

    def test_evaluation_workers_cannot_exceed_the_test_tasks(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(subprocess.CalledProcessError),
        ):
            prepared(directory=directory, extra=("--evaluation-workers", "11"))

    def test_worker_contract_defaults_to_v8(self):
        self.assertEqual(worker.contract(manifest={"runs": []}), (160000, 1))

    def test_worker_checks_each_row_against_the_manifest(self):
        def parsed(*, budget=65000, workers=4, rng="per-task"):
            return argparse.Namespace(
                practice_step_budget=budget, evaluation_workers=workers, evaluation_task_rng=rng
            )

        worker.check_budget_and_workers(parsed=parsed(), manifest=ITERATION)
        worker.check_budget_and_workers(
            parsed=parsed(budget=160000, workers=1, rng="shared"), manifest={}
        )
        for bad, manifest in (
            (parsed(), {}),
            (parsed(budget=160000), ITERATION),
            (parsed(workers=1), ITERATION),
            (parsed(rng="shared"), ITERATION),
        ):
            with self.subTest(parsed=bad, manifest=manifest), self.assertRaises(ValueError):
                worker.check_budget_and_workers(parsed=bad, manifest=manifest)

    def test_smoke_evaluates_as_many_tasks_as_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            launch = Path(directory) / "launch"
            row = {"arm": "ees", "seed": 0, "argv": []}
            for manifest, tasks in (({}, "1"), (ITERATION, "4")):
                with self.subTest(tasks=tasks):
                    with (
                        patch.object(
                            worker.subprocess, "run", side_effect=RuntimeError("intercepted")
                        ) as run,
                        self.assertRaisesRegex(RuntimeError, "intercepted"),
                    ):
                        worker.run(launch=launch, row=row, smoke=True, manifest=manifest)
                    command = run.call_args.args[0]
                    self.assertEqual(command[command.index("--num-test-tasks") + 1], tasks)
                    (launch.parent / "smoke/ees/0/worker.log").unlink()
                    (launch.parent / "smoke/ees/0").rmdir()

    def test_full_run_must_reach_the_manifest_budget(self):
        for manifest, steps, passes in ((ITERATION, 65000, True), ({}, 65000, False)):
            with self.subTest(manifest=manifest), tempfile.TemporaryDirectory() as directory:
                launch = Path(directory) / "launch"
                launch.mkdir()
                for name in ("manifest.json", "snapshot.json", "runtime-lock.json"):
                    (launch / name).write_text("{}")
                row = {"arm": "ees", "seed": 0, "argv": []}
                with patch.object(
                    worker.subprocess, "run", side_effect=completed_run(practice_steps=steps)
                ):
                    if passes:
                        worker.run(launch=launch, row=row, smoke=False, manifest=manifest)
                    else:
                        with self.assertRaisesRegex(ValueError, "160000"):
                            worker.run(launch=launch, row=row, smoke=False, manifest=manifest)

    def test_a_repeated_stop_limit_reaches_every_row(self):
        with tempfile.TemporaryDirectory() as directory:
            output = prepared(
                directory=directory, extra=(*V9_STYLE_ARGS, "--repeated-stop-limit", "100")
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["repeated_stop_limit"], 100)
            for row in manifest["runs"]:
                argv = row["argv"]
                self.assertIn("--no-early-stop", argv)
                self.assertEqual(argv[argv.index("--repeated-stop-limit") + 1], "100")
            self.assertEqual(
                (output / "commands.txt").read_text().count("--repeated-stop-limit 100"), 9
            )

    def test_no_repeated_stop_limit_leaves_rows_without_the_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            output = prepared(directory=directory, extra=V9_STYLE_ARGS)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertNotIn("repeated_stop_limit", manifest)
            for row in manifest["runs"]:
                self.assertNotIn("--repeated-stop-limit", row["argv"])

    def test_a_repeated_stop_limit_must_be_positive(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(subprocess.CalledProcessError),
        ):
            prepared(directory=directory, extra=("--repeated-stop-limit", "0"))

    def test_worker_checks_each_rows_repeated_stop_limit(self):
        worker.check_repeated_stop_limit(
            parsed=argparse.Namespace(repeated_stop_limit=None), manifest={}
        )
        worker.check_repeated_stop_limit(
            parsed=argparse.Namespace(repeated_stop_limit=100),
            manifest={"repeated_stop_limit": 100},
        )
        for limit, manifest in (
            (100, {}),
            (None, {"repeated_stop_limit": 100}),
            (10, {"repeated_stop_limit": 100}),
        ):
            with self.subTest(limit=limit, manifest=manifest), self.assertRaises(ValueError):
                worker.check_repeated_stop_limit(
                    parsed=argparse.Namespace(repeated_stop_limit=limit), manifest=manifest
                )

    def test_a_repeated_stop_completes_a_run_only_under_a_limit(self):
        limited = {**ITERATION, "repeated_stop_limit": 100}
        for manifest, endpoint, steps, passes in (
            (limited, "repeated_stop", 4100, True),
            (limited, "practice_step_budget", 65000, True),
            (ITERATION, "repeated_stop", 4100, False),
            (limited, "repeated_zero_step", 4100, False),
        ):
            with (
                self.subTest(manifest=manifest, endpoint=endpoint),
                tempfile.TemporaryDirectory() as directory,
            ):
                launch = Path(directory) / "launch"
                launch.mkdir()
                for name in ("manifest.json", "snapshot.json", "runtime-lock.json"):
                    (launch / name).write_text("{}")
                row = {"arm": "expectimax", "seed": 0, "argv": []}
                with patch.object(
                    worker.subprocess,
                    "run",
                    side_effect=completed_run(practice_steps=steps, endpoint=endpoint),
                ):
                    if passes:
                        receipt = worker.run(launch=launch, row=row, smoke=False, manifest=manifest)
                        self.assertEqual(receipt["endpoint"]["reason"], endpoint)
                    else:
                        with self.assertRaises(ValueError):
                            worker.run(launch=launch, row=row, smoke=False, manifest=manifest)

    def test_a_reset_cost_sweep_expands_rows_per_cost(self):
        costs = ("5", "10", "20", "100", "10000", "0")
        with tempfile.TemporaryDirectory() as directory:
            output = prepared(
                directory=directory,
                extra=("--human-reset-costs", *costs, "--max-parallel", "54"),
            )
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["human_reset_costs"], [5, 10, 20, 100, 10000, 0])
            rows = manifest["runs"]
            self.assertEqual(
                [(row["human_reset_cost"], row["seed"], row["arm"]) for row in rows],
                # Seed-major, then cost in the order given, then arm: one seed's full
                # results arrive first.
                [
                    (cost, seed, arm)
                    for seed in range(3)
                    for cost in (5, 10, 20, 100, 10000, 0)
                    for arm in prepare.ARMS
                ],
            )
            self.assertEqual([row["index"] for row in rows], list(range(54)))
            self.assertEqual(len({row["output"] for row in rows}), 54)
            for row in rows:
                cost, argv = row["human_reset_cost"], row["argv"]
                self.assertEqual(row["output"], f"reset-cost-{cost}/{row['arm']}/{row['seed']}")
                # The cost lives only in the row's cost file; no flag prices a reset.
                self.assertNotIn("--human-reset-practice-cost", argv)
                path = argv[argv.index("--practice-cost-config") + 1]
                self.assertEqual(
                    path, f"{manifest['remote_root']}/launch/costs/human-reset-cost-{cost}.json"
                )
                local = output / "costs" / f"human-reset-cost-{cost}.json"
                self.assertEqual(
                    hashlib.sha256(local.read_bytes()).hexdigest(), row["costs_sha256"]
                )
                written = json.loads(local.read_text())
                self.assertEqual(
                    {
                        name: skill["cost"]["value"]
                        for name, skill in written["human_skills"].items()
                    },
                    {"ResetCubeBinSameSide": cost, "ResetCubeBinOppositeSide": cost},
                )
                base = json.loads((HERE / "costs.json").read_text())
                for key in ("robot_step", "human_weight", "objective_lambda"):
                    self.assertEqual(written[key], base[key])
            self.assertIn("--array=0-53%54\n", (output / "full.sbatch").read_text())
            self.assertIn(
                "/results/reset-cost-10000/curtis/2", (output / "commands.txt").read_text()
            )

    def test_a_reset_cost_sweep_refuses_duplicate_costs(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(subprocess.CalledProcessError),
        ):
            prepared(directory=directory, extra=("--human-reset-costs", "5", "5"))

    def test_constraint_reaches_both_sbatch_scripts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = prepared(directory=directory, extra=("--constraint", "genoa"))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["resources"]["constraint"], "genoa")
            for phase in ("smoke", "full"):
                self.assertIn(
                    "#SBATCH --constraint=genoa\n", (output / f"{phase}.sbatch").read_text()
                )

    def test_worker_expects_the_manifests_rows(self):
        self.assertEqual(
            worker.expected_rows(manifest={"runs": []}),
            [(None, seed, arm) for seed in range(3) for arm in worker.ARMS],
        )
        self.assertEqual(
            worker.expected_rows(manifest={"human_reset_costs": [5, 0]})[3:7],
            [(0, 0, "ees"), (0, 0, "expectimax"), (0, 0, "curtis"), (5, 1, "ees")],
        )

    def test_worker_checks_each_rows_reset_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "costs.json"
            costs = json.loads((HERE / "costs.json").read_text())
            for skill in costs["human_skills"].values():
                skill["cost"]["value"] = 10000
            path.write_text(json.dumps(costs))
            row = {
                "human_reset_cost": 10000,
                "costs_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

            def parsed():
                return argparse.Namespace(practice_cost_config=str(path))

            worker.check_row_costs(parsed=parsed(), row=row)
            worker.check_row_costs(
                parsed=argparse.Namespace(practice_cost_config=str(HERE / "costs.json")),
                row={},
            )
            for bad_parsed, bad_row in (
                (parsed(), {**row, "human_reset_cost": 5}),
                (parsed(), {**row, "costs_sha256": "0" * 64}),
                (parsed(), {}),
            ):
                with self.subTest(row=bad_row), self.assertRaises(ValueError):
                    worker.check_row_costs(parsed=bad_parsed, row=bad_row)

    def test_a_swept_rows_output_carries_its_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            launch = Path(directory) / "launch"
            row = {"arm": "ees", "seed": 0, "argv": [], "output": "reset-cost-10/ees/0"}
            for smoke, parent in ((True, "smoke"), (False, "results")):
                with (
                    self.subTest(smoke=smoke),
                    patch.object(
                        worker.subprocess, "run", side_effect=RuntimeError("intercepted")
                    ) as run,
                    self.assertRaisesRegex(RuntimeError, "intercepted"),
                ):
                    worker.run(launch=launch, row=row, smoke=smoke, manifest={})
                command = run.call_args.args[0]
                self.assertEqual(
                    command[command.index("--output-dir") + 1],
                    str(launch.parent / parent / "reset-cost-10/ees/0"),
                )

    def test_cost_contract_is_the_two_resets_at_five(self):
        worker.check_costs(costs=json.loads((HERE / "costs.json").read_text()))
        old = {
            "human_skills": {
                name: {"cost": {"value": 12}, "duration": {"value": 1}}
                for name in ("Recover_cube_control_near", "RelocateBinNear", "RestoreBinTarget")
            }
        }
        with self.assertRaisesRegex(ValueError, "human reset"):
            worker.check_costs(costs=old)
        repriced = json.loads((HERE / "costs.json").read_text())
        repriced["human_skills"]["ResetCubeBinSameSide"]["cost"]["value"] = 12
        with self.assertRaisesRegex(ValueError, "human reset"):
            worker.check_costs(costs=repriced)

    def test_reject_old_experiment_tag(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [
                sys.executable,
                str(HERE / "prepare.py"),
                "--out",
                directory,
                "--tag",
                "t3-ours-sidefree-20261030-0912",
            ]
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
