"""A control tick must not pay for a goal check nobody reads."""

import pytest

pytest.importorskip("kinder")

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment


def test_a_control_tick_does_not_evaluate_the_goal():
    env = KitchenEnvironment(requested_count=1)
    try:
        c = env.controller()
        calls = []
        real = c.scene._check_goals
        c.scene._check_goals = lambda: calls.append(1) or real()
        c.motor_step(target_q=c.data.qpos[c.arm_q].copy(), grip=c.grip)
        assert calls == []
        # The goal itself is still answered on request, by the same rule.
        assert c.scene._check_goals() == real()
        assert calls == [1]
    finally:
        env.close()
