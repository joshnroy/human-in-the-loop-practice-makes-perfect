"""Cycle-safe process restart preserves physics, learning, RNGs and evaluation history."""

import copy
import json
import random
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.step_protocol import (
    CHECKPOINT_FILENAME,
    DeploymentSnapshot,
    PracticeCheckpoint,
    StepPracticeRunner,
    StepProtocolResume,
)
from tests.test_step_protocol import FakeProblem, FakeStepProtocol, _step_args


class Simulation(FakeProblem):
    def __init__(self):
        super().__init__()
        self._rng = np.random.default_rng(42)
        self.env._rng = np.random.default_rng(19)
        self.physics = np.array([1.123456789012345, 0.0])
        self.trajectory = []

    def take_action(self, *, action):
        state = super().take_action(action=action)
        self.physics += [
            self._rng.random() + self.env._rng.random() + np.random.random() + random.random(),
            float(action[0]),
        ]
        self.trajectory.append(self.physics.tolist())
        return state


class Domain(FakeStepProtocol):
    simulation_checkpoints = True

    @staticmethod
    def checkpoint_practice_environment(*, problem):
        return copy.deepcopy((problem.state, problem.physics, problem.trajectory))

    @staticmethod
    def restore_checkpoint_environment(*, problem, checkpoint):
        problem.state, problem.physics, problem.trajectory = copy.deepcopy(checkpoint)

    @staticmethod
    def run_saved(**kwargs):
        pass


class Method:
    skill_provider = None

    def __init__(self, *, env):
        self.env = env
        self._rng = np.random.default_rng(12)
        self.updates = 0
        self.costs = []

    def configure_practice_accounting(self, *, accounting):
        self.accounting = accounting

    def practice_complete(self):
        return False

    def get_practice_policy(self, *, task):
        return lambda state: SimpleNamespace(
            action=np.array([float(self._rng.integers(1, 3))]), label="skill"
        )

    def observe_practice_action_budget(self, **kwargs):
        pass

    def observe_execution_cost(self, **kwargs):
        self.costs.append(kwargs)

    def end_cycle(self):
        self.updates += 1


class Evaluations:
    """A genuinely unfinished future until checkpointing asks for its result."""

    def __init__(self):
        self.submitted = []

    def submit(self, *, fn, **kwargs):
        self.submitted.append(kwargs["checkpoint"])
        return Deferred()

    def finish(self):
        pass

    def abort(self):
        pass


class Deferred(Future):
    def result(self, timeout=None):  # noqa: PLR0917 -- Future API
        if not self.done():
            self.set_result({
                "complete": True,
                "tasks": [{"task": 0, "goal": "goal", "solved": False}],
                "num_solved": 0,
                "num_total": 1,
            })
        return super().result(timeout)


@pytest.mark.parametrize("interrupt", [False, True], ids=["clean-pause", "process-interruption"])
def test_resumed_simulation_continues_identically_without_duplicate_eval0(
    *, tmp_path, monkeypatch, interrupt
):
    monkeypatch.setattr(
        DeploymentSnapshot,
        "encode",
        staticmethod(lambda *, method, **_: str((method.updates, method.costs)).encode()),
    )

    def run(*, output, stop=None, resume=False):
        random.seed(23)
        np.random.seed(24)
        problem = Simulation()
        method = Method(env=problem.env)
        args = _step_args(
            output_dir=output,
            budget=8,
            interval=100,
            checkpoint_cycles=True,
            max_steps_per_interaction=1,
            stop_after_cycles=stop,
            measure_after_learning_updates=[1, 2],
            resume=output if resume else None,
        )
        live = PracticeCheckpoint.live_objects(problem=problem, skill_provider=None)
        saved = (
            PracticeCheckpoint.load(path=output / CHECKPOINT_FILENAME, live=live)
            if resume
            else None
        )
        method = method if saved is None else saved["method"]
        pool = Evaluations()
        metrics = StepPracticeRunner.run(
            args=args,
            method=method,
            problem=problem,
            environment=Domain,
            evaluations=pool,
            saved=saved,
        )
        return problem, method, metrics, pool

    alone, alone_method, metrics, _ = run(output=tmp_path / "alone")
    if interrupt:
        original = PracticeCheckpoint.save

        def stopping(**kwargs):
            original(**kwargs)
            if kwargs["header"]["next_cycle"] == 1:
                raise KeyboardInterrupt

        with monkeypatch.context() as patch:
            patch.setattr(PracticeCheckpoint, "save", staticmethod(stopping))
            with pytest.raises(KeyboardInterrupt):
                run(output=tmp_path / "resumed", stop=1)
        first_pool = None
    else:
        paused, _, _, first_pool = run(output=tmp_path / "resumed", stop=1)
        assert len(paused.trajectory) == 1
    header = PracticeCheckpoint.header(path=tmp_path / "resumed" / CHECKPOINT_FILENAME)
    assert header["next_cycle"] == 1
    resumed, method, resumed_metrics, second_pool = run(output=tmp_path / "resumed", resume=True)
    assert resumed.trajectory == alone.trajectory
    assert method.updates == alone_method.updates
    assert method.costs == alone_method.costs
    assert resumed_metrics == metrics
    if first_pool is not None:
        assert first_pool.submitted == [0, 1]
    assert second_pool.submitted == [2, 3]
    rows = json.loads((tmp_path / "resumed" / "step_evaluations.json").read_text())
    assert [r["index"] for r in rows] == list(range(4))
    assert all(r["complete"] for r in rows)


def test_pause_is_operational_resume_flag():
    assert StepProtocolResume.comparable(
        arguments={"stop_after_cycles": "1"}
    ) == StepProtocolResume.comparable(arguments={"stop_after_cycles": "None"})


def test_engine_checkpoint_restores_float64_fingers_controls_and_solver_state():
    """Tiny contact model: no KINDER, assets, renderer, planner or heavy simulation."""
    import mujoco

    from hitl_pmp.environments.tossing3d.simulation_checkpoint import capture_engine, restore_engine

    xml = """<mujoco><option timestep="0.002"/><worldbody>
    <geom type="plane" size="1 1 .1"/><body pos="0 0 .12">
    <joint name="finger" type="slide" axis="0 0 1"/>
    <geom type="sphere" size=".1" mass="1"/></body></worldbody>
    <actuator><motor joint="finger"/></actuator></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[0] = 0.0123456789012345
    data.qvel[0] = -0.123456789012345
    data.ctrl[0] = 0.333333333333333
    data.qfrc_applied[0] = 0.123456789012345
    for _ in range(25):
        mujoco.mj_step(model, data)
    checkpoint = capture_engine(model=model, data=data)
    expected = []
    for _ in range(20):
        mujoco.mj_step(model, data)
        expected.append(capture_engine(model=model, data=data)["integration"])
    fresh_model = mujoco.MjModel.from_xml_string(xml)
    fresh_data = mujoco.MjData(fresh_model)
    restore_engine(model=fresh_model, data=fresh_data, checkpoint=checkpoint)
    np.testing.assert_array_equal(
        capture_engine(model=fresh_model, data=fresh_data)["integration"], checkpoint["integration"]
    )
    for state in expected:
        mujoco.mj_step(fresh_model, fresh_data)
        np.testing.assert_array_equal(
            capture_engine(model=fresh_model, data=fresh_data)["integration"], state
        )
    fresh_model.body_mass[1] += 0.1
    with pytest.raises(ValueError, match="fingerprint"):
        restore_engine(model=fresh_model, data=fresh_data, checkpoint=checkpoint)
    incompatible = {**checkpoint, "mujoco_version": "incompatible"}
    with pytest.raises(ValueError, match="version"):
        restore_engine(model=model, data=data, checkpoint=incompatible)


def test_kitchen_checkpoint_keeps_seeded_placement_counter(*, monkeypatch):
    from hitl_pmp.environments.kitchen_recycling3d.backend import KitchenKinderBackend
    from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend

    monkeypatch.setattr(KinderBackend, "simulation_checkpoint", lambda self: {"seed": 10})

    def rebuild(self, *, checkpoint):  # noqa: PLR0917 -- patched instance method
        self._placements = 0
        self.reset_rejections = []

    monkeypatch.setattr(KinderBackend, "restore_simulation_checkpoint", rebuild)
    backend = KitchenKinderBackend()
    backend._placements = 7
    backend.reset_rejections = [0, 2, 1]
    checkpoint = backend.simulation_checkpoint()
    restored = KitchenKinderBackend()
    restored.restore_simulation_checkpoint(checkpoint=checkpoint)
    assert restored._placements == 7
    assert restored.reset_rejections == [0, 2, 1]


def test_unsupported_simulation_refuses_checkpoint_before_building(*, tmp_path):
    from hitl_pmp.step_protocol import StepProtocolEntry

    with pytest.raises(ValueError, match="backend cannot resume"):
        StepProtocolEntry.run(
            args=_step_args(output_dir=tmp_path, budget=8, interval=100, checkpoint_cycles=True),
            method_cli=None,
            env_cli=None,
            environment=FakeStepProtocol,
        )
