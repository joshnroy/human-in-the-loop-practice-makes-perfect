"""Pins `choose_practice_target`'s exact observable output -- ranking, decision log and
competence-model bookkeeping -- against a fixture recorded from the implementation that
rebuilt `skill_costs()` once per scored candidate. Pricing the base table once per
ranking is a speed change only; any drift here means it changed what EES decides."""

import json
import math
from pathlib import Path
from typing import Any, Literal

import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

FIXTURE = Path(__file__).parent / "fixtures" / "ees_practice_target_ranking.json"


class RankingTrace:
    """Drives a seeded LightSwitch EES through three rankings with outcomes and a
    cycle boundary between them, and records everything a ranking can observably
    change."""

    @staticmethod
    def describe(*, ground_skill: GroundSkill) -> str:
        return f"{ground_skill.skill.name}({','.join(obj.name for obj in ground_skill.objects)})"

    @staticmethod
    def record(*, groundings: Literal["observed", "all"], log: Path) -> dict[str, Any]:
        env = LightSwitchEnvironment(grid_size=4)
        method = EesMethod(
            env=env,
            skill_provider=LightSwitchSkillProvider(env=env),
            seed=0,
            practice_groundings=groundings,
            ees_decision_log=log,
        )
        tasks = LightSwitchTasks(env=env, seed=0)
        for _ in range(2):
            task = tasks.sample_train_task()
            method.record_seen_task(
                init_atoms=method.abstract_state(state=task.initial_state),
                goal=task.goal.atoms,
            )
        skills = list(method.all_practice_groundings())
        rankings: list[list[str]] = []
        for round_index in range(3):
            for offset, ground in enumerate(skills[round_index::5]):
                method.observe_outcome(ground_skill=ground, success=(offset + round_index) % 3 != 0)
            rankings.append([
                RankingTrace.describe(ground_skill=ground)
                for ground in method.choose_practice_target()
            ])
            method.advance_competence_cycle()
        return {
            "rankings": rankings,
            "competence_models": [
                RankingTrace.describe(ground_skill=ground) for ground in method._competence_models
            ],
            "current_competences": [
                [RankingTrace.describe(ground_skill=ground), competence]
                for ground, competence in method.current_competences().items()
            ],
            "skill_costs": [
                [RankingTrace.describe(ground_skill=ground), cost]
                for ground, cost in method.skill_costs().items()
            ],
            "decision_log": [json.loads(line) for line in log.read_text().splitlines()],
        }


@pytest.mark.parametrize("groundings", ["observed", "all"])
def test_practice_target_ranking_matches_the_per_candidate_pricing_fixture(
    *, groundings: Literal["observed", "all"], tmp_path: Path
) -> None:
    trace = RankingTrace.record(groundings=groundings, log=tmp_path / "decisions.jsonl")
    expected = json.loads(FIXTURE.read_text())[groundings]
    assert trace["rankings"] == expected["rankings"]
    assert trace["competence_models"] == expected["competence_models"]
    assert trace["current_competences"] == expected["current_competences"]
    assert trace["skill_costs"] == expected["skill_costs"]
    assert trace["decision_log"] == expected["decision_log"]
    scores = [record for record in trace["decision_log"] if record["event"] == "score"]
    assert scores and all(math.isfinite(record["score"]) for record in scores)
