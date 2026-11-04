"""The kitchen pick's waypoint stepping on an arm that tracks every command exactly.

Upstream commands kp = 2 position deltas. On an arm that tracks them, the error to a
waypoint goes e -> e - 2e = -e: a period-2 cycle that never enters a 0.04 rad
acceptance band once |e| exceeds it. Kitchen v8's step-limit aborts sat in that cycle
around an intermediate lift waypoint until the cube dropped.
"""

import importlib.util
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder_models") is None, reason="needs KINDER"
)


class ExactArm:
    """A controller stepped on an arm that moves by exactly the commanded delta."""

    @staticmethod
    def controller(*, cls: type, plan: list[list[float]]) -> Any:
        controller = cls.__new__(cls)
        phase = cls.PickCubeControllerPhase.LIFT_CUBE_TO_HOME
        controller.plans = {phase: plan}
        controller._plan_step_idx = {phase: 0}
        controller.current_phase = phase
        controller._lifted = False
        controller.ACTION_DIM = 11
        controller.conf = np.array(plan[0][:7], dtype=float)
        controller._get_current_robot_arm_conf = lambda: list(controller.conf)
        controller._get_current_robot_gripper_pose = lambda: 1.0
        return controller

    @staticmethod
    def errors(*, controller: Any, ticks: int) -> list[np.ndarray]:
        """Per tick, the error to the waypoint being tracked, until the plan is done."""
        phase = controller.current_phase
        plan = controller.plans[phase]
        errors = []
        for _ in range(ticks):
            if controller._lifted:
                break
            action = controller._step_trajectory_phase(phase, None)
            controller.conf = controller.conf + np.asarray(action[3:10], dtype=float)
            target = np.asarray(plan[controller._plan_step_idx[phase]][:7])
            errors.append(target - controller.conf)
        return errors

    @staticmethod
    def plan() -> list[list[float]]:
        """Start, an intermediate waypoint 0.046 rad away on the wrist (as in v8's
        curtis seed 0 lift), then an endpoint further on."""
        start = [0.0, 0.3, 0.0, 1.5, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        waypoint = list(start)
        waypoint[6] += 0.046
        endpoint = list(waypoint)
        endpoint[6] += 0.3
        return [start, waypoint, endpoint]


def test_upstreams_stepping_cycles_around_an_intermediate_waypoint():
    """The premise: Tossing3D's (upstream's) pick, unchanged, never reaches it."""
    from kinder_models.dynamic3d.tossing.parameterized_skills import PickCubeController

    controller = ExactArm.controller(cls=PickCubeController, plan=ExactArm.plan())
    errors = ExactArm.errors(controller=controller, ticks=60)
    assert not controller._lifted
    assert controller._plan_step_idx[controller.current_phase] == 1
    wrist = [e[6] for e in errors[-10:]]
    assert all(a * b < 0 for a, b in zip(wrist[:-1], wrist[1:], strict=True))


def test_the_kitchen_pick_damps_a_sign_flip_and_passes_the_waypoint():
    from hitl_pmp.environments.kitchen_recycling3d.controllers import RecyclingPickController

    controller = ExactArm.controller(cls=RecyclingPickController, plan=ExactArm.plan())
    errors = ExactArm.errors(controller=controller, ticks=60)
    assert controller._lifted, len(errors)
    flips = sum(
        int(np.any((np.sign(a) * np.sign(b) < 0) & (np.abs(a) > 1e-6)))
        for a, b in zip(errors[:-1], errors[1:], strict=True)
    )
    assert flips <= 2, flips
    assert len(errors) < 15


def test_the_kitchen_pick_leaves_a_monotone_approach_undamped():
    """No sign flip, no damping: the command is upstream's kp = 2 delta, so motion that
    converges today keeps its timing."""
    from kinder_models.dynamic3d.tossing.parameterized_skills import PickCubeController

    from hitl_pmp.environments.kitchen_recycling3d.controllers import RecyclingPickController

    plan = ExactArm.plan()
    plan[1][6] = plan[0][6] + 0.5
    plan[2][6] = plan[1][6] + 0.5
    kitchen = ExactArm.controller(cls=RecyclingPickController, plan=plan)
    upstream = ExactArm.controller(cls=PickCubeController, plan=plan)
    phase = kitchen.current_phase
    for _ in range(4):
        a = kitchen._step_trajectory_phase(phase, None)
        b = upstream._step_trajectory_phase(phase, None)
        np.testing.assert_array_equal(a, b)
        for controller, action in ((kitchen, a), (upstream, b)):
            controller.conf = controller.conf + np.asarray(action[3:10], dtype=float)
