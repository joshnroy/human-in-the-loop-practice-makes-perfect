"""The always-ask reset wrapper end to end on the live Tossing3D simulator: one short
practice period through the real `PracticeLoop`, where every toss -- the one skill with
continuous parameters -- is followed by one full reset to a freshly sampled start
state, granted and charged like any other human reset."""

import argparse
import importlib.util

import pytest

from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.practice_makes_perfect.always_ask_reset import (
    PARAMETERIZED_SKILL,
    AlwaysAskReset,
)
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.practice_loop import PracticeLoop

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator dependency"
)


def test_every_toss_is_followed_by_one_human_reset() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    Tossing3DCli.add_arguments(parser=parser)
    problem = Tossing3DCli.build_practice_problem(args=parser.parse_args(["--no-scene-bg"]))
    rule = AlwaysAskReset(cost=5.0)
    method = EesMethod(
        env=problem.env,
        skill_provider=Tossing3DSkillProvider(env=problem.env),
        always_ask_reset=rule,
        # EXP-22c's EES settings, under which the toss is practised early; with EES's
        # defaults a short period can practise only the pick, wrapped or not.
        reset_cost_gate=False,
        goal_pursuit_horizon=0,
        reproduce_predicators_explore_target_only=True,
        reproduce_predicators_random_when_stranded=True,
        reproduce_predicators_seen_task_order=True,
        reproduce_predicators_skip_perfect=True,
    )
    tosses: list[str] = []
    original = EesMethod.record_action_cost

    def counting(self, *, ground_skill) -> None:  # noqa: PLR0917 - stands in for a bound method
        if ground_skill.skill.name == "MoveToTossLocationAndToss":
            tosses.append(ground_skill.skill.name)
        original(self, ground_skill=ground_skill)

    metrics = Metrics()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(EesMethod, "record_action_cost", counting)
        try:
            PracticeLoop.run(
                problem=problem,
                method=method,
                metrics=metrics,
                num_cycles=1,
                max_steps_per_interaction=12,
                num_test_tasks=0,
            )
        finally:
            problem.env.close()

    assert rule.skill_attempts == len(tosses) > 0
    assert rule.forced.get(PARAMETERIZED_SKILL, 0) == rule.skill_attempts - int(rule.owed)
    _, interventions = metrics.num_human_interventions()
    assert interventions >= sum(rule.forced.values())
