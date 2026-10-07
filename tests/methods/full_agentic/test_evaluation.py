"""Held-out execution stops at the native goal or common physical horizon."""

import pytest

from hitl_pmp.full_agentic.runner import ScoreBridge


class FakeBridge:
    def __init__(self, *, goal_after):
        self.calls = 0
        self.goal_after = goal_after
        self.env = self

    def backend(self):
        return self

    def check_goals(self):
        return self.calls >= self.goal_after

    def step(self, *, action):
        self.calls += 1
        return {"calls": self.calls}


@pytest.mark.parametrize(("goal_after", "expected_solved"), [(2, True), (20, False)])
def test_no_physics_after_goal_or_horizon(*, goal_after, expected_solved):
    physical = FakeBridge(goal_after=goal_after)
    bridge = ScoreBridge(bridge=physical, budget=2)
    for _ in range(2):
        bridge.step(action=[0] * 18)
    with pytest.raises(RuntimeError, match="episode_finished"):
        bridge.step(action=[0] * 18)
    assert physical.calls == bridge.clock.steps == 2
    assert bridge.success == expected_solved
