"""EES-always-reset end to end on the live Tossing3D simulator.

One short practice period through the real `PracticeLoop`: every reset the rule asks
for is granted as a full reset (a new scene seed, `steps_taken` back to 0, the robot
rebuilt with it), and the resets granted equal the practice-target executions.
"""

import argparse
import importlib.util

import pytest

from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.methods.practice_makes_perfect.practice_target_reset import (
    AlwaysResetAfterPracticeTarget,
)
from hitl_pmp.practice_loop import PracticeLoop

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator dependency"
)


def test_every_target_execution_is_followed_by_one_full_reset(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    Tossing3DCli.add_arguments(parser=parser)
    problem = Tossing3DCli.build_practice_problem(args=parser.parse_args(["--no-scene-bg"]))
    provider = Tossing3DSkillProvider(env=problem.env)
    method = EesMethod(
        env=problem.env,
        skill_provider=provider,
        practice_target_reset=AlwaysResetAfterPracticeTarget.for_provider(skill_provider=provider),
        reset_cost_gate=False,
        # The canonical runs' setting: no goal phase, so the period is all practice.
        goal_pursuit_horizon=0,
    )
    scene = Tossing3DEnvironment.scene
    scenes: list[tuple[int, int, int]] = []
    original = Problem.execute_sampled_start_reset

    def recording(self: Problem) -> None:  # noqa: PLR0917 - stands in for a bound method
        before = int(round(self.get_current_state().get(obj=scene, feature_name="seed")))
        original(self)
        after = self.get_current_state()
        scenes.append((
            before,
            int(round(after.get(obj=scene, feature_name="seed"))),
            int(round(after.get(obj=scene, feature_name="steps_taken"))),
        ))

    monkeypatch.setattr(Problem, "execute_sampled_start_reset", recording)
    try:
        PracticeLoop.run(
            problem=problem,
            method=method,
            metrics=Metrics(),
            num_cycles=1,
            max_steps_per_interaction=6,
            num_test_tasks=0,
        )
    finally:
        problem.env.close()

    rule = method.practice_target_reset
    assert rule is not None
    assert rule.target_executions > 0
    assert len(scenes) == rule.requests == rule.target_executions - int(rule.owed)
    for before, after, steps_taken in scenes:
        assert after != before
        assert steps_taken == 0
