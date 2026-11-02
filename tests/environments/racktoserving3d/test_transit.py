from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.transit import ArmTransit
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed


def make_tool():
    return SimpleNamespace(
        shelf=SimpleNamespace(
            model=SimpleNamespace(
                jnt_limited=np.ones(2, dtype=bool),
                jnt_range=np.array([[-np.pi, np.pi], [-np.pi, np.pi]]),
            ),
            joints=[0, 1],
            home_arm=np.zeros(2),
        ),
        check_configuration=Mock(),
        check_arm_segment=Mock(),
    )


def test_clear_transit_returns_checked_goal_without_mutating_start():
    tool = make_tool()
    planner = ArmTransit(tool=tool, base=np.zeros(3), holding=True)
    start, target = np.zeros(2), np.ones(2)
    route = planner.plan(start=start, target=target)
    assert route is not None
    np.testing.assert_allclose(route, [target])
    np.testing.assert_array_equal(start, np.zeros(2))
    assert planner.edge_checks == 1
    assert tool.check_configuration.call_count == 2
    assert tool.check_arm_segment.call_args.kwargs["holding"]


def test_failed_transit_respects_edge_budget():
    tool = make_tool()
    tool.check_arm_segment.side_effect = SkillFailed("blocked")
    planner = ArmTransit(tool=tool, base=np.zeros(3), holding=False, max_edge_checks=1)
    assert planner.plan(start=np.zeros(2), target=np.ones(2)) is None
    assert planner.edge_checks == 1
    tool.check_arm_segment.assert_called_once()


def test_transit_rejects_invalid_joint_goal():
    planner = ArmTransit(tool=make_tool(), base=np.zeros(3), holding=False)
    with pytest.raises(SkillFailed, match="goal violates joint limits"):
        planner.plan(start=np.zeros(2), target=np.array([4.0, 0]))


def test_wrapped_shortcut_cannot_cross_joint_stop():
    tool = make_tool()
    planner = ArmTransit(tool=tool, base=np.zeros(3), holding=False)
    planner.deadline = float("inf")
    assert not planner.edge_clear(start=np.array([3.0, 0]), target=np.array([-3.0, 0]))
    tool.check_arm_segment.assert_not_called()
