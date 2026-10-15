"""The Full-Agentic shelf controller gets raw 11-D actuators and numeric state, not an
oracle: no skills, predicates, goal labels or resets cross the bridge."""

import importlib.util
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from gymnasium.spaces import Box
from relational_structs import Object, ObjectCentricState, Type

from hitl_pmp.environments.sweepoffshelf3d.agentic_bridge import SweepOffShelf3DAgenticBridge
from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment

HAS_KINDER = importlib.util.find_spec("kinder") is not None
TOOL = {
    "pos_tool_x": 0.2,
    "pos_tool_y": -0.3,
    "pos_tool_z": 0.7,
    "quat_tool_w": 1.0,
    "quat_tool_x": 0.0,
    "quat_tool_y": 0.0,
    "quat_tool_z": 0.0,
}


def scene() -> ObjectCentricState:
    movable = Type("mujoco_movable_object")
    collider = Type("mujoco_static_collider")
    robot = Type("mujoco_tidybot_robot")
    movable_features = [
        "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz",
        "bb_x", "bb_y", "bb_z",
    ]  # fmt: skip
    return ObjectCentricState(
        {
            Object("cube_0", movable): np.arange(16, dtype=float),
            Object("collider:cupboard_1:9", collider): np.arange(10, dtype=float),
            Object("robot", robot): np.zeros(1),
        },
        {
            movable: movable_features,
            collider: ["x", "y", "z", "qw", "qx", "qy", "qz", "bb_x", "bb_y", "bb_z"],
            robot: ["pos_base_x"],
        },
    )


@pytest.fixture
def bridge(*, monkeypatch: pytest.MonkeyPatch) -> SweepOffShelf3DAgenticBridge:
    robot = SimpleNamespace(
        act_delta=True,
        control_frequency=10.0,
        qpos={"base": [0.0] * 3, "arm": [0.1] * 7, "gripper": [0.0, 0.0]},
        qvel={"base": [0.0] * 3, "arm": [0.0] * 7, "gripper": [0.0, 0.0]},
        ctrl={"gripper": [255.0]},
        sim=SimpleNamespace(data=SimpleNamespace(mj_data=SimpleNamespace(time=3.0))),
    )
    gym = Mock()
    gym.action_space = Box(
        np.array([-0.1] * 10 + [0.0]), np.array([0.1] * 10 + [1.0]), dtype=np.float64
    )
    backend = SimpleNamespace(env=lambda: gym, grip=0.0)
    env = SweepOffShelf3DEnvironment()
    env._backend = backend  # noqa: SLF001 -- a scene without MuJoCo
    state = scene()
    monkeypatch.setattr(SweepOffShelf3DAgenticBridge, "_robot", lambda self: robot)
    monkeypatch.setattr(SweepOffShelf3DAgenticBridge, "_state", lambda self: state)
    monkeypatch.setattr(
        SweepOffShelf3DAgenticBridge, "_tool_pose", lambda self, *, proprioception: TOOL
    )
    return SweepOffShelf3DAgenticBridge(env=env, step_limit=2)


def gym_of(*, bridge: SweepOffShelf3DAgenticBridge) -> Mock:
    return bridge.env.backend().env()


class TestShelfAgenticBridge:
    def test_object_state_is_numeric_with_no_task_labels(
        self, *, bridge: SweepOffShelf3DAgenticBridge
    ) -> None:
        observation = bridge.observe()
        assert set(observation) == {
            "observation_mode",
            "proprioception",
            "control_step",
            "simulation_time_s",
            "action_spec",
            "objects",
            "state_spec",
        }
        assert observation["observation_mode"] == "object_state"
        assert observation["simulation_time_s"] == 3.0
        proprioception = observation["proprioception"]
        assert len(proprioception) == 23 + len(TOOL)
        assert proprioception["gripper_command"] == 1.0
        assert proprioception["pos_arm_joint7"] == 0.1
        objects = {o["name"]: o for o in observation["objects"]}
        assert set(objects) == {"cube_0", "collider:cupboard_1:9", "robot"}
        assert objects["cube_0"]["features"]["bb_z"] == 15.0
        assert "vx" not in objects["collider:cupboard_1:9"]["features"]
        assert objects["robot"]["features"] == proprioception
        gym_of(bridge=bridge).step.assert_not_called()

    def test_a_label_feature_never_crosses_the_bridge(
        self, *, bridge: SweepOffShelf3DAgenticBridge, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        labelled = Type("labelled")
        state = ObjectCentricState(
            {Object("cube_0", labelled): np.array([1.0, 0.0])}, {labelled: ["x", "in_box"]}
        )
        monkeypatch.setattr(SweepOffShelf3DAgenticBridge, "_state", lambda self: state)
        with pytest.raises(ValueError, match="unsupported physical object features"):
            bridge.observe()

    def test_action_spec_is_the_native_11d_delta_command_without_schedules(
        self, *, bridge: SweepOffShelf3DAgenticBridge
    ) -> None:
        spec = bridge.action_spec()
        assert spec["shape"] == [11]
        assert spec["low"] == [-0.1] * 10 + [0.0]
        assert spec["high"] == [0.1] * 10 + [1.0]
        assert spec["gripper"] == {"index": 10, "open": 0.0, "closed": 1.0}
        assert spec["arm_position"]["indices"] == list(range(3, 10))
        assert not {"schedule_rows", "schedule_timestep_s", "arm_velocity"} & spec.keys()

    def test_a_valid_command_advances_one_period_and_records_the_grip(
        self, *, bridge: SweepOffShelf3DAgenticBridge
    ) -> None:
        command = [0.05] * 10 + [1.0]
        bridge.step(action={"values": command})
        (sent,), _ = gym_of(bridge=bridge).step.call_args
        assert sent.tolist() == command
        assert bridge.control_steps == 1
        assert bridge.env.backend().grip == 1.0

    @pytest.mark.parametrize(
        "action",
        [
            [0.0] * 18,
            [[0.0] * 11] * 100,
            {"schedule": [0.0] * 11},
            [0.2] + [0.0] * 10,
            [0.0] * 10 + [float("nan")],
        ],
    )
    def test_an_invalid_command_never_reaches_physics(
        self, *, bridge: SweepOffShelf3DAgenticBridge, action: object
    ) -> None:
        with pytest.raises(ValueError):
            bridge.step(action=action)
        gym_of(bridge=bridge).step.assert_not_called()
        assert bridge.control_steps == 0

    def test_the_step_limit_is_enforced(self, *, bridge: SweepOffShelf3DAgenticBridge) -> None:
        for _ in range(2):
            bridge.step(action=[0.0] * 11)
        with pytest.raises(RuntimeError, match="budget"):
            bridge.step(action=[0.0] * 11)
        assert gym_of(bridge=bridge).step.call_count == 2


@pytest.mark.skipif(not HAS_KINDER, reason="needs KINDER")
@pytest.mark.xdist_group("sweepoffshelf3d_agentic_bridge_sim")
class TestShelfAgenticBridgeLive:
    def test_live_scene_observation_kinematics_actions_and_human_reset(self) -> None:
        from hitl_pmp.full_agentic.domains import PRACTICE_STEP_LIMIT, SweepOffShelf3DFullAgentic

        env = SweepOffShelf3DFullAgentic.make_environment(options={}, human_seed=0)
        try:
            env.hard_reset()
            bridge = SweepOffShelf3DFullAgentic.make_bridge(env=env, step_limit=PRACTICE_STEP_LIMIT)
            backend = env.backend()
            observation = bridge.observe()
            names = {o["name"] for o in observation["objects"]}
            assert {"robot", "box_0", "wiper_0", "cupboard_1"} | {
                f"cube_{i}" for i in range(5)
            } <= names
            assert observation["action_spec"]["shape"] == [11]
            chain = bridge.robot_spec()["kinematic_chain"]
            assert sum(len(link["joints"]) for link in chain["links"]) == 10
            site = backend.model().site("robot_pinch_site").id
            tool = [observation["proprioception"][f"pos_tool_{a}"] for a in "xyz"]
            assert np.allclose(tool, backend.data().site_xpos[site], atol=1e-4)

            before = observation["simulation_time_s"]
            after = bridge.step(action=[0.0] * 10 + [1.0])
            assert after["simulation_time_s"] == pytest.approx(before + 0.1)
            assert after["proprioception"]["gripper_command"] == pytest.approx(1.0)
            assert not SweepOffShelf3DFullAgentic.goal_check(env=env)()

            # The human's reset is seen at once, with no robot step in between.
            cube = backend.core().get_object("cube_0")
            cube.set_pose([0.5, 0.5, 0.05], [1.0, 0.0, 0.0, 0.0])
            backend.sim().forward()
            assert env.reset_movables(destination="reset_cubes")
            moved = {o["name"]: o for o in bridge.observe()["objects"]}["cube_0"]
            assert moved["features"]["x"] != pytest.approx(0.5)
        finally:
            env.close()
