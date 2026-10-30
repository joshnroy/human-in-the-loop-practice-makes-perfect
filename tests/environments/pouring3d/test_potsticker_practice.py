"""Production dumpling configuration, count isolation, and interruption contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.cli import Cli
from hitl_pmp.core.control_steps import ControlStepLimitReached
from hitl_pmp.environments.pouring3d.feedback_controller import FeedbackPourParameters
from hitl_pmp.environments.pouring3d.potsticker_controller import PotstickerController
from hitl_pmp.environments.pouring3d.practice.cli import FrozenPourCli
from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment
from hitl_pmp.environments.pouring3d.practice.skill_provider import FrozenPourSkillProvider


def test_default_cli_uses_three_dumplings_and_independent_skills():
    args = Cli.parse_args(argv=["--env", "pouring3d", "--method", "ees"])
    problem = FrozenPourCli.build_problem(args=args)
    assert problem.env.payload_kind == "potsticker"
    provider = FrozenPourSkillProvider(env=problem.env)
    pours = [s for s in provider.skills() if s.name.startswith("Pour")]
    assert [s.name for s in pours] == ["Pour1", "Pour2", "Pour3"]
    assert all(s.param_dim == 3 for s in pours)
    assert [o.name for o in problem.env.objects().shrimp()] == [f"potsticker_{i}" for i in range(3)]
    assert len({provider.ground(name=f"Pour{i}", request=i) for i in range(1, 4)}) == 3


@pytest.mark.parametrize(
    "kwargs", [{"payload_count": 5}, {"layout": "floor"}, {"receiving_vessel": "tray"}]
)
def test_dumpling_configuration_rejects_unvalidated_variants(*, kwargs):
    with pytest.raises(ValueError, match="three pieces and station skillet"):
        FrozenPourEnvironment(payload_kind="potsticker", **kwargs)


def test_joint_pour_restores_test_goal_on_interruption():
    def interrupt(**_):
        raise ControlStepLimitReached("test")

    fake = SimpleNamespace(
        held="pitcher",
        scene=SimpleNamespace(requested_count=1),
        prepare_to_pour=lambda: None,
        pour_prepared=interrupt,
    )
    with pytest.raises(ControlStepLimitReached):
        PotstickerController.run_joint_pour(
            fake, parameters=FeedbackPourParameters(), requested_count=2
        )
    assert fake.scene.requested_count == 1
    assert fake.mode == "joint_lateral" and fake.anticipation_seconds == 0


def test_unreachable_preparation_tries_alternate_grasp_without_reversing_unexecuted_path(
    *, monkeypatch
):
    from hitl_pmp.environments.pouring3d.practice.controller import PracticePourController

    attempts = []

    def pick(self, *, index, location):  # noqa: PLR0917 -- patched bound method
        attempts.append(self.recovery_pitch_degrees)
        if len(attempts) == 1:
            raise ValueError("No collision-free empty-arm preparation path: reachable poses 0")
        self.held = "potsticker_2"

    def unexpected_retreat(self):  # noqa: PLR0917 -- patched bound method
        raise AssertionError("No arm approach was executed to reverse")

    monkeypatch.setattr(PracticePourController, "pick_shrimp", pick)
    monkeypatch.setattr(PracticePourController, "clear_failed_pick", unexpected_retreat)
    c = PotstickerController.model_construct(
        scene=SimpleNamespace(cube_locations=lambda: ["pitcher", "pitcher", "tray"]), held=None
    )
    c.pick_payload_with_retry(index=2, location="tray")
    assert attempts == [60, 30]
    assert c.held == "potsticker_2" and c.recovery_pitch_degrees == 60


def test_failed_mobile_approach_retraces_measured_base_and_arm(*, monkeypatch):
    from hitl_pmp.environments.pouring3d.feedback_controller import FeedbackPourController
    from hitl_pmp.environments.pouring3d.practice.controller import PracticePourController

    moves = []

    def step(self, *, target_q, grip):  # noqa: PLR0917 -- patched bound method
        self.data.qpos[self.base_q] = self.base_target
        self.data.qpos[self.arm_q] = target_q
        self.steps += 1
        moves.append(self.data.qpos.copy())

    monkeypatch.setattr(FeedbackPourController, "motor_step", step)
    monkeypatch.setattr(PracticePourController, "pause", lambda self, **kw: None)
    c = PracticePourController.model_construct(
        data=SimpleNamespace(qpos=np.zeros(10)),
        base_q=np.arange(3),
        arm_q=np.arange(3, 10),
        base_target=np.zeros(3),
        last_q=np.zeros(7),
        held=None,
        steps=0,
        shrimp_recovery_active=True,
        phase="position for shrimp pick",
    )
    for value in (0.1, 0.2):
        c.base_target = np.full(3, value)
        c.motor_step(target_q=np.full(7, value), grip=0)
    c.clear_failed_pick()
    assert np.allclose(c.data.qpos, 0)
    assert any(np.allclose(move, 0.1) for move in moves[2:])
    assert not c.shrimp_approach_path


@pytest.mark.parametrize("contained", [False, True])
def test_premature_drop_accepts_only_measured_containment(*, monkeypatch, contained):
    from hitl_pmp.environments.pouring3d.practice.controller import PracticePourController

    def drop(self, *, index):  # noqa: PLR0917 -- patched bound method
        self.held = None
        self.phase = "center held shrimp over pitcher"
        raise ValueError("Physical shrimp grasp lost")

    pauses = []
    monkeypatch.setattr(PracticePourController, "drop_shrimp", drop)
    monkeypatch.setattr(PotstickerController, "pause", lambda self, **kw: pauses.append(kw))
    monkeypatch.setattr(PotstickerController, "move", lambda self, **kw: None)
    monkeypatch.setattr(
        PotstickerController, "hand_pose", lambda self: (np.zeros(3), Rotation.identity())
    )
    c = PotstickerController.model_construct(
        scene=SimpleNamespace(
            cube_names=["potsticker_0"],
            cube_in_vessel=lambda *args: contained,
            cubes_settled=lambda: True,
        ),
        held="potsticker_0",
    )
    if not contained:
        with pytest.raises(ValueError, match="Physical shrimp grasp lost"):
            c.drop_shrimp(index=0)
        assert not pauses
    else:
        c.drop_shrimp(index=0)
        assert c.events == [{"event": "contained release before command", "payload": 0}]
        assert pauses[0]["grip"] == 0
        assert sum(p["seconds"] for p in pauses) == pytest.approx(4)


def test_native_dumpling_reset_and_recording(*, tmp_path):
    import argparse
    import json

    import mujoco
    import numpy as np

    from hitl_pmp.environments.pouring3d.practice.problem import FrozenPourProblem
    from hitl_pmp.environments.pouring3d.practice.recording import FrozenPourReplayRenderer
    from hitl_pmp.environments.pouring3d.practice.step_protocol import FrozenPourStepProtocol
    from hitl_pmp.environments.pouring3d.practice.tasks import FrozenPourTasks
    from hitl_pmp.recording.run_record.types import Phase

    env = FrozenPourEnvironment(payload_kind="potsticker", scene_bg=False)
    replay = None
    writer = None
    try:
        c = env.controller()
        assert c.ready() and c.mode == "joint_lateral"
        assert c.env.spec.id == "kinder/FrozenPotstickerPour3D-o3-v0"
        obj = c.scene.get_object("potsticker_0")
        obj.set_pose([0.68, -0.85, 0.741], [1, 0, 0, 0])
        obj.set_velocity([0.0] * 3, [0.0] * 3)
        mujoco.mj_forward(c.model, c.data)
        q = c.data.qpos.copy()
        v = c.data.qvel.copy()
        time = c.data.time
        assert env.reset_movables(destination="reset_shrimp")
        keepq = np.ones(c.model.nq, dtype=bool)
        keepv = np.ones(c.model.nv, dtype=bool)
        for name in c.scene.cube_names:
            j = c.model.body_jntadr[c.model.body(name).id]
            a = c.model.jnt_qposadr[j]
            b = c.model.jnt_dofadr[j]
            keepq[a : a + 7] = False
            keepv[b : b + 6] = False
        np.testing.assert_array_equal(q[keepq], c.data.qpos[keepq])
        np.testing.assert_array_equal(v[keepv], c.data.qvel[keepv])
        assert c.data.time == time and c.ready()
        path = tmp_path / "recording.jsonl"
        problem = FrozenPourProblem(env=env, tasks=FrozenPourTasks(env=env))
        writer = FrozenPourStepProtocol.attach_recording(
            args=argparse.Namespace(record_motion="all", method="test", seed=0),
            problem=problem,
            output_path=path,
        )
        writer.begin_segment(phase=Phase.PRACTICE)
        env.take_action(action=env.noop_action())
        writer.close()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        header = rows[0]
        # Header field structure is the shared run-record format.
        scene = header["scene"]
        assert scene["payload_kind"] == "potsticker" and scene["practice_model_version"] == 6
        replay = FrozenPourReplayRenderer(scene=scene, scene_bg=False, width=320, height=240)
        live = FrozenPourStepProtocol.live_state(env=env)
        assert replay.show(state=live).shape == (240, 320, 3)
        assert replay.current_state() == live
    finally:
        if writer is not None:
            writer.close()
        if replay is not None:
            replay.close()
        env.close()


def test_approach_planning_restores_controller_state_on_failure():
    import numpy as np
    from scipy.spatial.transform import Rotation

    class Probe:
        last_q = np.arange(7, dtype=float)
        posture = 1.0
        phase = "above payload"
        base_target = np.zeros(3)
        base_q = np.arange(3)
        data = SimpleNamespace(qpos=np.zeros(10))
        payload_approach_plan = []
        calls = 0

        def inverse_kinematics(self, **_):
            self.calls += 1
            self.last_q = self.last_q + 0.01
            self.posture += 0.02
            raise ValueError("IK infeasible in approach")

    probe = Probe()
    before = probe.last_q.copy()
    assert not PotstickerController.payload_approach_feasible(
        probe,
        q=np.zeros(7),
        preparation=np.array([0.4, 1.0, 1.1]),
        grasp=np.array([0.7, 1.0, 0.9]),
        rotation=Rotation.identity(),
        base=np.array([0.2, 1.0, 0]),
    )
    np.testing.assert_array_equal(probe.last_q, before)
    np.testing.assert_array_equal(probe.base_target, np.zeros(3))
    np.testing.assert_array_equal(probe.data.qpos, np.zeros(10))
    assert probe.posture == 1.0 and probe.phase == "above payload" and probe.calls == 1


def test_retention_correction_applies_to_carry_hold_command(*, monkeypatch):
    import numpy as np

    from hitl_pmp.environments.pouring3d.native_controller import NativePouringController
    from hitl_pmp.environments.pouring3d.practice.controller import PracticePourController

    commands = []

    def motor(self, *, target_q, grip):  # noqa: PLR0917 -- bound-method test replacement
        commands.append(grip)
        self.steps += 1

    monkeypatch.setattr(NativePouringController, "motor_step", motor)
    monkeypatch.setattr(
        PracticePourController, "finger_contacts", lambda self, **_: {"left", "right"}
    )
    c = PracticePourController.model_construct(
        scene=SimpleNamespace(cube_names=("potsticker_0",)),
        held="potsticker_0",
        recovery_contact_close=True,
        payload_hold_grip=0.5,
        lost_contacts=1,
    )
    c.motor_step(target_q=np.zeros(7), grip=0.5)
    assert commands == [pytest.approx(0.53)]
    assert c.held == "potsticker_0" and c.lost_contacts == 0
