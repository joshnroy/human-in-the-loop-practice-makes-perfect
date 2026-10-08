import json
import tempfile
import unittest
from pathlib import Path

from hitl_pmp.full_agentic.world import World


class FakeBridge:
    def __init__(self):
        self.calls = 0
        self.env = self
        self.helps = 0
        self.fail = False

    def observe(self):
        return {"calls": self.calls, "helps": self.helps}

    def action_spec(self):
        return {"low": [-1] * 18, "high": [1] * 18, "schedule_rows": 10}

    def step(self, *, action):
        self.calls += 1
        if self.fail:
            raise RuntimeError("partial physical execution")
        return self.observe()

    def reset_movables(self, *, destination):
        self.helps += 1
        return True


class WorldTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        workspace = self.root / "workspace"
        workspace.mkdir()
        (workspace / "approach.py").write_text("class GeneratedApproach: pass")
        self.bridge = FakeBridge()
        self.world = World(
            bridge=self.bridge, workspace=workspace, output=self.root, deadline=float("inf")
        )
        self.counter = 0

    def tearDown(self):
        self.world.db.close()
        self.tmp.cleanup()

    def request(self, op, **kwargs):  # noqa: PLR0917 -- test request builder
        self.counter += 1
        return dict(
            operation=op,
            request_id=str(self.counter),
            expected_version=self.world.version,
            **kwargs,
        )

    def test_duplicate_action_and_human_request_execute_once(self):
        self.world.dispatch(request=self.request("begin_trial"))
        step = self.request("step", action=[0] * 18)
        response = self.world.dispatch(request=step)
        self.assertEqual(response, self.world.dispatch(request=step))
        self.assertEqual(self.bridge.calls, 1)
        self.world.dispatch(request=self.request("end_trial"))
        help_request = self.request("request_help", intervention_id="reset_cube_far")
        response = self.world.dispatch(request=help_request)
        self.assertEqual(response, self.world.dispatch(request=help_request))
        self.assertEqual(self.bridge.helps, 1)
        self.assertEqual(self.world.observation()["accumulated_cost"], 2)

    def test_direct_steps_and_help_require_no_trial_boundaries(self):
        self.world.configure_measurements(budget=2000, interval=2000)
        for _ in range(1001):
            response = self.world.dispatch(request=self.request("step", action=[0] * 18))
            self.assertIn("result", response)
        self.world.dispatch(request=self.request("request_help", intervention_id="reset_cube_far"))
        self.world.dispatch(request=self.request("finish_adaptation"))
        self.assertEqual(self.world.counted_steps, 1002)
        self.assertEqual(self.world.observation()["accumulated_cost"], 1002)

    def test_failed_human_attempt_is_priced_once_with_its_own_duration(self):
        from hitl_pmp.core.practice_costs import ChargeFunction, HumanCharge, PracticeCosts

        self.world.configure_measurements(
            budget=10,
            interval=3,
            costs=PracticeCosts(
                human_skills={
                    "reset_cube_and_bin_near": HumanCharge(
                        cost=ChargeFunction(value=7), duration=ChargeFunction(value=3)
                    )
                },
                human_weight=2,
            ),
        )
        self.bridge.reset_movables = lambda **kwargs: False
        request = self.request("request_help", intervention_id="reset_cube_and_bin_near")
        response = self.world.dispatch(request=request)
        self.assertIn("error", response)
        self.assertEqual(response, self.world.dispatch(request=request))
        self.assertEqual(self.world.counted_steps, 3)
        self.assertEqual(self.world.observation()["accumulated_cost"], 14)
        self.assertEqual(self.world.measurements[-1]["human_cost"], 7)
        self.assertEqual(self.world.human_by_side["robot_side"], 1)

    def test_initial_measurement_needs_no_supplied_controller(self):
        (self.world.workspace / "approach.py").unlink()
        self.world.configure_measurements(budget=10, interval=2)
        self.assertFalse(self.world.measurements[0]["controller_present"])
        self.assertEqual(self.world.measurements[0]["physical_cost"], 0)

    def test_stale_request_and_id_collision_rejected(self):
        start = self.request("begin_trial")
        self.world.dispatch(request=start)
        with self.assertRaises(ValueError):
            self.world.dispatch(request=dict(start, operation="step", action=[0] * 18))
        with self.assertRaises(ValueError):
            self.world.dispatch(
                request=dict(self.request("step", action=[0] * 18), expected_version=0)
            )
        self.assertEqual(self.bridge.calls, 0)

    def test_partial_execution_blocks_further_actions_and_can_be_retrieved(self):
        self.world.dispatch(request=self.request("begin_trial"))
        self.bridge.fail = True
        step = self.request("step", action=[0] * 18)
        response = self.world.dispatch(request=step)
        self.assertIn("error", response)
        self.assertEqual(response, self.world.dispatch(request=step))
        with self.assertRaises(RuntimeError):
            self.world.dispatch(request=self.request("step", action=[0] * 18))
        self.assertEqual(self.bridge.calls, 1)

    def test_submission_freezes_code_but_does_not_reset_world(self):
        self.world.dispatch(request=self.request("finish_adaptation"))
        (self.world.workspace / "approach.py").write_text("changed")
        self.assertEqual(
            (self.world.submission / "approach.py").read_text(), "class GeneratedApproach: pass"
        )
        with self.assertRaises(RuntimeError):
            self.world.dispatch(request=self.request("begin_trial"))
        self.assertEqual(self.bridge.helps, 0)

    def test_invalid_command_does_not_advance_or_poison_world(self):
        self.world.dispatch(request=self.request("begin_trial"))
        with self.assertRaises(ValueError):
            self.world.dispatch(request=self.request("step", action=[10] * 18))
        self.assertFalse(self.world.uncertain)
        self.assertEqual(self.bridge.calls, 0)
        result = self.world.dispatch(request=self.request("step", action=[0] * 18))
        self.assertIn("result", result)

    def test_no_free_reset_and_pending_receipt_is_not_replayed(self):
        with self.assertRaises(ValueError):
            self.world.dispatch(request=self.request("reset"))
        req = self.request("request_help", intervention_id="reset_cube_far")
        self.world.db.execute(
            "INSERT INTO receipts VALUES(?,?,NULL)",
            (req["request_id"], json.dumps(req, sort_keys=True)),
        )
        self.world.db.commit()
        with self.assertRaises(RuntimeError):
            self.world.dispatch(request=req)
        self.assertEqual(self.bridge.helps, 0)


class MeasurementTests(WorldTests):
    def test_measurements_count_help_without_ending_trial_or_learning(self):
        self.world.configure_measurements(budget=5, interval=2, human_steps=1)
        self.world.dispatch(request=self.request("begin_trial"))
        self.world.dispatch(request=self.request("step", action=[0] * 18))
        self.world.dispatch(request=self.request("step", action=[0] * 18))
        self.assertTrue(self.world.active)
        self.assertEqual([m["practice_steps"] for m in self.world.measurements], [0, 2])
        frozen = self.world.measurements[-1]["snapshot"]
        (self.world.workspace / "approach.py").write_text("changed")
        self.assertEqual(
            (Path(frozen) / "approach.py").read_text(), "class GeneratedApproach: pass"
        )
        self.world.dispatch(request=self.request("end_trial"))
        self.world.dispatch(request=self.request("request_help", intervention_id="reset_cube_far"))
        self.assertEqual(self.world.counted_steps, 3)
        self.assertEqual(self.world.human_by_side, {"opposite_side": 1, "robot_side": 0})
        self.world.dispatch(request=self.request("begin_trial"))
        for _ in range(2):
            self.world.dispatch(request=self.request("step", action=[0] * 18))
        with self.assertRaises(RuntimeError):
            self.world.dispatch(request=self.request("step", action=[0] * 18))
        self.assertEqual(self.bridge.calls, 4)
        self.world.dispatch(request=self.request("end_trial"))
        self.world.dispatch(request=self.request("finish_adaptation"))
        self.assertEqual([m["practice_steps"] for m in self.world.measurements], [0, 2, 4, 5])

    def test_duplicate_human_receipt_is_not_double_counted(self):
        self.world.configure_measurements(budget=10, interval=1, human_steps=1)
        request = self.request("request_help", intervention_id="reset_cube_and_bin_near")
        self.world.dispatch(request=request)
        self.world.dispatch(request=request)
        self.assertEqual(self.world.counted_steps, 1)
        self.assertEqual(len(self.world.measurements), 2)

    def test_early_endpoint_has_actual_count_not_projected_measurements(self):
        self.world.configure_measurements(budget=10, interval=4, human_steps=1)
        self.world.dispatch(request=self.request("begin_trial"))
        self.world.dispatch(request=self.request("step", action=[0] * 18))
        self.world.dispatch(request=self.request("end_trial"))
        self.world.dispatch(request=self.request("finish_adaptation"))
        self.assertEqual([m["practice_steps"] for m in self.world.measurements], [0, 1])

    def test_final_code_update_at_same_step_has_its_own_measurement(self):
        self.world.configure_measurements(budget=10, interval=1)
        self.world.dispatch(request=self.request("begin_trial"))
        self.world.dispatch(request=self.request("step", action=[0] * 18))
        self.world.dispatch(request=self.request("end_trial"))
        (self.world.workspace / "approach.py").write_text("class Updated: pass")
        self.world.dispatch(request=self.request("finish_adaptation"))
        self.assertEqual([m["practice_steps"] for m in self.world.measurements], [0, 1, 1])
        self.assertNotEqual(
            self.world.measurements[-2]["sha256"], self.world.measurements[-1]["sha256"]
        )

    def test_pending_evaluation_does_not_block_or_reset_practice(self):
        from concurrent.futures import Future

        pending = []

        def enqueue(record):  # noqa: PLR0917 -- callback
            future = Future()
            pending.append((record, future))

        self.world.configure_measurements(budget=10, interval=1, callback=enqueue)
        self.world.dispatch(request=self.request("begin_trial"))
        for _ in range(2):
            self.world.dispatch(request=self.request("step", action=[0] * 18))
        self.assertEqual(self.bridge.calls, 2)
        self.assertEqual(self.bridge.helps, 0)
        self.assertTrue(self.world.active)
        self.assertEqual(len(pending), 3)
        self.assertTrue(all(not f.done() for _, f in pending))
