"""Offline packaging safety checks; no simulator, SSH or scheduler calls."""

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
