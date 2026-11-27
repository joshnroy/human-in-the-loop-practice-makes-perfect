"""A robot skill that refuses (0 controller steps, state unchanged) is not applicable
from that state, so EES must not choose it again there.

Della sweep EES p=0.5 seed 0 (2026-11-06): in a stuck kitchen state every robot skill
refused with 0 ticks, and EES spent five whole sessions dispatching the same refusing
skill, because a refusal was scored as an ordinary failed attempt whose forecast gain
stayed the top practice score."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from hitl_pmp.core.method.method import InteractionComplete
from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.environments.lightswitch.skills import LightSwitchSkills
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod, _EesEpisode


class _OwnSelector(EesMethod):
    """Stands in for DP and the belief-space planners: each overrides
    `select_skill_to_practice`, which is what scopes EES-only behaviour."""

    def select_skill_to_practice(self, *, true_atoms):
        raise NotImplementedError


def _setup(*, monkeypatch, tmp_path: Path, method_class: type[EesMethod] = EesMethod):
    env = LightSwitchEnvironment(grid_size=4)
    method = method_class(
        env=env,
        skill_provider=LightSwitchSkillProvider(env=env),
        num_candidates=4,
        max_proposals_per_candidate=3,
    )
    on = GroundSkill(
        skill=LightSwitchSkills.TURN_ON_LIGHT,
        objects=(env.robot, env.get_cells()[-1], env.light),
    )
    off = GroundSkill(skill=LightSwitchSkills.TURN_OFF_LIGHT, objects=on.objects)
    # A fixed abstract state: nothing the skills do changes it, as in a refusal.
    atoms = frozenset(on.preconditions | off.preconditions)
    monkeypatch.setattr(method_class, "abstract_state", lambda self, **kwargs: atoms)
    monkeypatch.setattr(
        method_class, "select_skill_to_practice", Mock(side_effect=lambda **kwargs: [on, off])
    )
    state = env.build_initial_state(light_level=0.2, light_target=0.8)
    return method, on, off, atoms, state


def test_a_zero_step_refusal_is_not_chosen_again_from_the_same_state(
    *, monkeypatch, tmp_path
) -> None:
    method, on, off, atoms, state = _setup(monkeypatch=monkeypatch, tmp_path=tmp_path)
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True)
    episode.step(state=state)
    assert episode._pending == on  # noqa: SLF001
    method.observe_execution_cost(cost=0.0, steps=0, complete=True)
    episode.step(state=state)
    assert episode._pending == off  # noqa: SLF001 (the refusing skill was deselected)
    assert on in method.starved_ground_skills(true_atoms=atoms)


def test_a_failure_that_ran_the_controller_stays_selectable(*, monkeypatch, tmp_path) -> None:
    """Only a refusal is evidence of inapplicability; a failed attempt that executed
    control steps is ordinary competence evidence, even if the state ends unchanged."""
    method, on, _off, atoms, state = _setup(monkeypatch=monkeypatch, tmp_path=tmp_path)
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True)
    episode.step(state=state)
    method.observe_execution_cost(cost=1.0, steps=40, complete=True)
    episode.step(state=state)
    assert episode._pending == on  # noqa: SLF001
    assert method.starved_ground_skills(true_atoms=atoms) == frozenset()


def test_a_refusal_still_counts_as_a_failed_attempt(*, monkeypatch, tmp_path) -> None:
    """Deselection only: what a refusal contributes to competence is unchanged."""
    method, on, _off, _atoms, state = _setup(monkeypatch=monkeypatch, tmp_path=tmp_path)
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True)
    episode.step(state=state)
    method.observe_execution_cost(cost=0.0, steps=0, complete=True)
    episode.step(state=state)
    assert method.practice_outcomes()[on.skill.name].num_attempts == 1


def test_when_every_robot_skill_refuses_the_period_ends(*, monkeypatch, tmp_path) -> None:
    method, on, off, atoms, state = _setup(monkeypatch=monkeypatch, tmp_path=tmp_path)
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True)
    episode.step(state=state)
    method.observe_execution_cost(cost=0.0, steps=0, complete=True)
    episode.step(state=state)
    method.observe_execution_cost(cost=0.0, steps=0, complete=True)
    with pytest.raises(InteractionComplete) as caught:
        episode.step(state=state)
    assert not caught.value.planner_stop
    assert method.starved_ground_skills(true_atoms=atoms) == frozenset({on, off})


def test_a_method_with_its_own_selector_is_unaffected(*, monkeypatch, tmp_path) -> None:
    """Scoped to EES's own selector: DP and the belief-space planners keep their
    behaviour."""
    method, on, _off, atoms, state = _setup(
        monkeypatch=monkeypatch, tmp_path=tmp_path, method_class=_OwnSelector
    )
    EesMethod.observe_execution_cost(method, cost=0.0, steps=0, complete=True)
    episode = _EesEpisode(method=method, goal=frozenset(), practicing=True)
    episode.step(state=state)
    EesMethod.observe_execution_cost(method, cost=0.0, steps=0, complete=True)
    episode.observe_pending(true_atoms=atoms, state=state)
    assert method.starved_ground_skills(true_atoms=atoms) == frozenset()
    del on
