"""EES can start each named skill's competence from its own Beta prior
(`skill_competence_priors`, from `--skill-prior` under
`--ees-competence-from-skill-prior`); every other skill keeps the shared prior.

Built on SweepOffShelf3D's symbolic layer only; nothing here touches the simulator.
"""

import math

import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.skill_provider import SweepOffShelf3DSkillProvider
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod


def _build(*, priors: dict[str, tuple[float, float]] | None = None) -> EesMethod:
    env = SweepOffShelf3DEnvironment()
    return EesMethod(
        env=env,
        skill_provider=SweepOffShelf3DSkillProvider(env=env),
        practice_groundings="all",
        **({} if priors is None else {"skill_competence_priors": priors}),
    )


def _robot_grounds(*, method: EesMethod) -> tuple[GroundSkill, ...]:
    return method.all_practice_groundings()


def test_without_skill_priors_every_skill_starts_at_the_shared_prior() -> None:
    method = _build()
    shared = method.prior_alpha / (method.prior_alpha + method.prior_beta)
    for ground in _robot_grounds(method=method):
        assert method.competence_model(ground_skill=ground).get_current_competence() == (
            pytest.approx(shared)
        )


def test_a_named_robot_skill_starts_at_its_own_prior_and_others_keep_the_shared_one() -> None:
    probe = _build()
    grounds = _robot_grounds(method=probe)
    named = grounds[0].skill.name
    method = _build(priors={named: (0.25, 2.5)})
    shared = method.prior_alpha / (method.prior_alpha + method.prior_beta)
    for ground in grounds:
        expected = 0.25 / 2.75 if ground.skill.name == named else shared
        assert method.competence_model(ground_skill=ground).get_current_competence() == (
            pytest.approx(expected)
        )


def test_a_named_human_skill_starts_at_its_own_prior() -> None:
    probe = _build()
    reset = probe.skill_provider.human_cube_bin_reset_skills()[0]
    method = _build(priors={reset.skill.name: (1.0, 1.0)})
    assert method.human_skill_competence(ground_skill=reset) == pytest.approx(0.5)


def test_an_untried_named_skill_is_planned_at_its_own_prior_cost() -> None:
    """A ground skill missing from the cost table is priced from its own prior, not
    the shared prior's -log mean."""
    probe = _build()
    ground = _robot_grounds(method=probe)[0]
    method = _build(priors={ground.skill.name: (1.0, 3.0)})
    assert method.prior_cost(ground_skill=ground) == pytest.approx(-math.log(0.25))
    other = next(g for g in _robot_grounds(method=method) if g.skill != ground.skill)
    assert method.prior_cost(ground_skill=other) == pytest.approx(method.default_cost())


def test_the_planner_prices_an_absent_named_skill_from_its_own_prior(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hitl_pmp.planning.fast_downward import FastDownwardPlanner

    probe = _build()
    ground = _robot_grounds(method=probe)[0]
    method = _build(priors={ground.skill.name: (1.0, 3.0)})
    captured: dict = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(FastDownwardPlanner, "plan", capture)
    method._plan_or_raise(  # noqa: SLF001
        skills=method.skills(), init_atoms=frozenset(), goal=frozenset(), ground_skill_costs={}
    )
    assert captured["ground_skill_costs"][ground] == pytest.approx(-math.log(0.25))


def test_a_skill_prior_must_be_a_proper_beta() -> None:
    with pytest.raises(ValueError, match="skill_competence_priors"):
        _build(priors={"anything": (0.0, 1.0)})
