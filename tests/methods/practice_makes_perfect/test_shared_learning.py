import json
import math

import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.practice_makes_perfect.ees_method import STOP_SKILL, EesMethod, _EesEpisode
from hitl_pmp.planning.fast_downward import PlanningFailure


def build(**kwargs):
    environment = Tossing3DEnvironment()
    return EesMethod(
        env=environment,
        skill_provider=Tossing3DSkillProvider(env=environment),
        use_ucb_bonus=False,
        reproduce_predicators_double_observe=False,
        **kwargs,
    )


def picks(*, method):
    """Use the still side-bound PickCube to exercise distinct-grounding sharing."""
    skill = next(skill for skill in method.skills() if skill.name == "PickCube")
    objects = {obj.name: obj for obj in method.objects()}
    return tuple(
        GroundSkill(
            skill=skill,
            objects=tuple(
                objects[name] for name in ("robot", "cube_0", "cuboid_barrier", side, "bin_0")
            ),
        )
        for side in ("robot_side", "opposite_side")
    )


def observe(*, method, ground, outcomes):
    for outcome in outcomes:
        method.observe_outcome(ground_skill=ground, success=outcome)


def test_default_forecasts_keep_grounded_histories():
    method = build()
    near, far = picks(method=method)
    observe(method=method, ground=near, outcomes=[True] * 8)
    observe(method=method, ground=far, outcomes=[False] * 4)
    assert method.competence_sharing == "grounded"
    assert method.practice_groundings == "observed"
    assert method.forecast_competence_model(ground_skill=near).get_current_competence() != (
        method.forecast_competence_model(ground_skill=far).get_current_competence()
    )


def test_parent_forecast_pools_actual_outcomes_once_with_one_prior():
    method = build(competence_sharing="parent")
    near, far = picks(method=method)
    observe(method=method, ground=near, outcomes=[True] * 8)
    observe(method=method, ground=far, outcomes=[False] * 4)
    pooled = method.forecast_competence_model(ground_skill=near)
    assert pooled is method.forecast_competence_model(ground_skill=far)
    assert pooled.num_observations == 12
    assert pooled.get_current_competence() == pytest.approx(18 / 23)
    assert method.total_observations() == 12
    assert method.skill_costs()[near] == method.skill_costs()[far]
    toss = next(skill for skill in method.skills() if skill.name == "MoveToTossLocationAndToss")
    assert method.sampler(skill_name=toss.name, param_dim=toss.param_dim) is method.sampler(
        skill_name=toss.name, param_dim=toss.param_dim
    )


def test_rate_only_shares_improvement_not_current_competence():
    method = build(competence_sharing="rate")
    near, far = picks(method=method)
    observe(method=method, ground=near, outcomes=[True] * 8)
    observe(method=method, ground=far, outcomes=[False] * 4)
    assert method.forecast_competence_model(ground_skill=near).get_current_competence() != (
        method.forecast_competence_model(ground_skill=far).get_current_competence()
    )
    assert method.forecast_competence_gain(ground_skill=near) == (
        method.forecast_competence_gain(ground_skill=far)
    )
    assert method.pooled_competence_model(ground_skill=far).num_observations == 12


@pytest.mark.parametrize("sharing", ["parent", "rate"])
def test_near_practice_reprices_far_pick_in_ees_task_plan(*, sharing, monkeypatch):
    method = build(competence_sharing=sharing)
    near, far = picks(method=method)
    observe(method=method, ground=near, outcomes=[False] * 8)
    method.competence_model(ground_skill=far)
    monkeypatch.setattr(EesMethod, "planning_progress_plans", lambda self, **kwargs: [[far]])
    current = method.skill_costs()[far]
    score = method.score_ground_skill(ground_skill=near)
    assert score > -current
    far_current = method.forecast_competence_model(ground_skill=far).get_current_competence()
    gain = method.forecast_competence_gain(ground_skill=near)
    assert score == pytest.approx(math.log(min(1.0, far_current + gain)))


def test_all_groundings_exposes_unseen_targets_without_changing_skills():
    original = build()
    flexible = build(practice_groundings="all")
    near, far = picks(method=flexible)
    assert flexible.skills() == original.skills()
    assert flexible.skill_provider.human_cube_bin_reset_skills() == (
        original.skill_provider.human_cube_bin_reset_skills()
    )
    assert original.practice_candidates() == []
    assert {near, far} <= set(flexible.practice_candidates())


@pytest.mark.parametrize("sharing", ["parent", "rate"])
def test_shared_ees_target_keeps_its_scored_grounding(*, sharing, monkeypatch):
    method = build(competence_sharing=sharing, practice_groundings="all")
    near, far = picks(method=method)
    monkeypatch.setattr(
        EesMethod, "select_skill_to_practice", lambda self, **kwargs: [far, STOP_SKILL]
    )
    planned_goals = []

    def plan(**kwargs):
        planned_goals.append(kwargs["goal"])
        return []

    monkeypatch.setattr(EesMethod, "plan_to", lambda self, **kwargs: plan(**kwargs))
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True)
    assert method.practice_alternatives(ground_skill=far) == [far]
    assert episode._practice_plan(true_atoms=near.preconditions) == [far]
    assert planned_goals == [far.preconditions]


@pytest.mark.parametrize("sharing", ["grounded", "parent", "rate"])
def test_all_ees_groundings_keep_paramless_open_gripper(*, sharing):
    method = build(competence_sharing=sharing, practice_groundings="all")
    candidates = method.practice_candidates()
    grippers = [ground for ground in candidates if ground.skill.name == "OpenGripper"]
    assert grippers
    assert all(ground.skill.param_dim == 0 for ground in grippers)
    assert set(candidates) == set(method.choose_practice_target())


def test_no_transfer_mode_does_not_substitute_an_unallocated_grounding():
    method = build(competence_sharing="grounded", practice_groundings="all")
    near, far = picks(method=method)
    assert method.practice_alternatives(ground_skill=far) == [far]
    assert near not in method.practice_alternatives(ground_skill=far)


def test_parent_evidence_excludes_random_attempts_and_advances_once():
    method = build(competence_sharing="parent")
    near, far = picks(method=method)
    observe(method=method, ground=near, outcomes=[True])
    method.observe_outcome(ground_skill=far, success=False, was_random_exploration=True)
    method.advance_competence_cycle()
    observe(method=method, ground=far, outcomes=[False])
    pooled = method.pooled_competence_model(ground_skill=near)
    assert pooled.cycle_observations == [[True], [False]]
    assert method.total_observations() == 2


def test_parent_mode_does_not_pool_human_reset_beliefs():
    method = build(competence_sharing="parent", practice_groundings="all")
    near, far = method.skill_provider.human_cube_bin_reset_skills()
    method.human_competence_model(ground_skill=near).observe(success=False)
    assert method.human_competence_model(ground_skill=far).num_observations == 0


@pytest.mark.parametrize("sharing", ["parent", "rate"])
def test_pricing_unseen_forecasts_does_not_expand_observed_practice_menu(*, sharing):
    method = build(competence_sharing=sharing)
    near, far = picks(method=method)
    observe(method=method, ground=near, outcomes=[True])
    assert far in method.skill_costs()
    assert method.practice_candidates() == [near]
    method.skill_costs()
    method.advance_competence_cycle()
    assert method.practice_candidates() == [near]
    assert method.total_observations() == 1


@pytest.mark.parametrize("sharing", ["parent", "rate"])
def test_forecast_only_priors_do_not_raise_the_observed_reset_ceiling(*, sharing, monkeypatch):
    environment = Tossing3DEnvironment()
    method = EesMethod(
        env=environment,
        skill_provider=Tossing3DSkillProvider(env=environment, human_reset_practice_cost=0.09),
        competence_sharing=sharing,
        score_human_skills=False,
        reset_cost_gate=True,
    )
    near, _far = picks(method=method)
    observe(method=method, ground=near, outcomes=[True])
    reset = method.offered_partial_resets()[0]
    calls = []

    def plan(*_arguments, **kwargs):
        calls.append(kwargs)
        return [reset] if len(calls) == 1 else []

    monkeypatch.setattr(EesMethod, "_plan_or_raise", plan)
    assert (
        method.plan_to(
            init_atoms=frozenset(), goal=frozenset(), costs=method.skill_costs(), practicing=True
        )
        == []
    )
    assert len(calls) == 2


@pytest.mark.parametrize("sharing", ["parent", "rate"])
def test_flexible_setup_compares_human_prices_and_handles_missing_alternatives(
    *, sharing, monkeypatch
):
    method = build(competence_sharing=sharing, practice_groundings="all")
    near, far = picks(method=method)
    near_reset, far_reset = method.offered_partial_resets()
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True)
    unavailable = set()

    def prefix(*_arguments, **kwargs):
        ground = near if kwargs["goal"] == near.preconditions else far
        if ground in unavailable:
            raise PlanningFailure("unreachable alternative")
        return [near_reset if ground == near else far_reset]

    monkeypatch.setattr(EesMethod, "plan_to", prefix)
    arguments = dict(alternatives=[near, far], true_atoms=frozenset(), starved=frozenset())
    assert episode._flexible_practice_plan(**arguments) == [near_reset, near]
    method.human_competence_model(ground_skill=near_reset).observe(success=False)
    assert episode._flexible_practice_plan(**arguments) == [far_reset, far]
    arguments["starved"] = frozenset({far})
    assert episode._flexible_practice_plan(**arguments) == [near_reset, near]
    arguments["starved"] = frozenset()
    unavailable.add(far)
    assert episode._flexible_practice_plan(**arguments) == [near_reset, near]
    unavailable.add(near)
    assert episode._flexible_practice_plan(**arguments) is None


def test_eager_and_lazy_pool_initialization_have_identical_evidence_and_cycles():
    eager = build(competence_sharing="parent")
    lazy = build(competence_sharing="parent")
    near, far = picks(method=eager)
    eager.pooled_competence_model(ground_skill=far)
    for method in (eager, lazy):
        method.observe_outcome(ground_skill=far, success=False, was_random_exploration=True)
        observe(method=method, ground=near, outcomes=[True])
        method.advance_competence_cycle()
        observe(method=method, ground=far, outcomes=[False, True])
    assert (
        eager.pooled_competence_model(ground_skill=near).cycle_observations
        == (lazy.pooled_competence_model(ground_skill=far).cycle_observations)
        == [[True], [False, True]]
    )


def test_forecast_log_preserves_ground_counts_and_shared_identity(*, tmp_path, monkeypatch):
    path = tmp_path / "forecasts.jsonl"
    method = build(competence_sharing="parent", learning_forecast_log=path)
    near, far = picks(method=method)
    observe(method=method, ground=near, outcomes=[True, True])
    observe(method=method, ground=far, outcomes=[False])
    monkeypatch.setattr(EesMethod, "fit_samplers", lambda self: None)
    monkeypatch.setattr(EesMethod, "refresh_planning_progress_plans", lambda self: None)
    method.end_cycle()
    record = json.loads(path.read_text())
    assert record["competence_sharing"] == "parent"
    assert [item["ground_observations"] for item in record["forecasts"]] == [2, 1]
    assert {item["pooled_observations"] for item in record["forecasts"]} == {3}
