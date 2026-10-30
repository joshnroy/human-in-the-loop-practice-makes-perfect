"""Production dumpling configuration, count isolation, and interruption contracts."""

from types import SimpleNamespace

import pytest

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
