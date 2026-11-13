"""`break_ties_among_best`: a tie at the top score that is not a tie over everything.

The learned sampler treats a best score shared by more than `uninformative_tie_fraction`
of the candidates as "the classifier cannot discriminate" and draws uniformly over ALL
candidates. That is right when every candidate ties (an unfitted or constant classifier)
and wrong when the classifier has saturated over a good region wider than half the
range: Serving's Sweep succeeds over about two thirds of its d range, so an accurate
sampler was treated as uninformative and deployed a d from the failing third about one
time in three. With the switch on, such a pick is drawn among the tied-best."""

import argparse
import pickle

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.environments.lightswitch.skills import LightSwitchSkills
from hitl_pmp.methods.practice_makes_perfect.cli import EesCli
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.methods.practice_makes_perfect.wrapped_sampler import LearnedSkillSampler
from hitl_pmp.step_protocol import DeploymentSnapshot

_CANDIDATES = [np.array([value]) for value in np.linspace(0.0, 1.0, 30)]
_ROWS = [[1.0, float(candidate[0])] for candidate in _CANDIDATES]


def _sampler(*, scores, seed=0):
    sampler = LearnedSkillSampler(skill_name="Sweep", param_dim=1, seed=seed)
    object.__setattr__(sampler, "score_inputs", lambda *, sampler_inputs: list(scores))
    return sampler


def _top_two_thirds():
    """The bottom third scores 0, the top two thirds tie at 1."""
    return [0.0] * 10 + [1.0] * 20


@pytest.mark.parametrize("explore", [False, True])
def test_a_tie_over_the_top_two_thirds_never_yields_the_bottom_third(*, explore):
    sampler = _sampler(scores=_top_two_thirds())
    object.__setattr__(sampler, "exploration_epsilon", 0.0)
    picked = set()
    for _ in range(300):
        choice = sampler.sample(
            sampler_inputs=_ROWS,
            candidates=_CANDIDATES,
            explore=explore,
            break_ties_among_best=True,
        )
        assert choice.was_informed and not choice.was_random
        assert choice.score == 1.0
        picked.add(float(choice.params[0]))
    assert min(picked) >= float(_CANDIDATES[10][0])
    # Uniform among the tied-best: over 300 draws every one of the 20 is picked.
    assert len(picked) == 20


def _draws(*, scores, count=50, **switch):
    sampler = _sampler(scores=scores, seed=7)
    choices = [
        sampler.sample(sampler_inputs=_ROWS, candidates=_CANDIDATES, explore=explore, **switch)
        for explore in [False, True] * (count // 2)
    ]
    return (
        [(float(c.params[0]), c.was_random, c.was_informed, c.score) for c in choices],
        sampler._rng.bit_generator.state,
    )


@pytest.mark.parametrize("score", [0.5, 0.0, 1.0])
def test_a_tie_over_every_candidate_is_drawn_exactly_as_before(*, score):
    scores = [score] * 30
    on = _draws(scores=scores, break_ties_among_best=True)
    assert on == _draws(scores=scores, break_ties_among_best=False)
    assert on == _draws(scores=scores)
    assert all(not informed and not random for _, random, informed, _ in on[0])


def test_with_the_switch_off_a_top_tie_is_still_drawn_over_all_candidates():
    """The behaviour every run so far had: a uniform index over all 30, from the
    sampler's own generator, reported as uninformed."""
    draws, state = _draws(scores=_top_two_thirds())
    twin = np.random.default_rng(7)
    expected = [float(_CANDIDATES[int(twin.integers(0, 30))][0]) for _ in range(50)]
    assert [value for value, _, _, _ in draws] == expected
    assert state == twin.bit_generator.state
    assert all(not informed and not random for _, random, informed, _ in draws)
    assert min(expected) < float(_CANDIDATES[10][0])
    assert _draws(scores=_top_two_thirds(), break_ties_among_best=False) == (draws, state)


def test_a_discriminating_batch_is_unchanged_by_the_switch():
    scores = list(np.linspace(0.0, 1.0, 30))
    assert _draws(scores=scores, break_ties_among_best=True) == _draws(scores=scores)


def _method(**extra):
    env = LightSwitchEnvironment(grid_size=3)
    return EesMethod(env=env, skill_provider=LightSwitchSkillProvider(env=env), seed=0, **extra)


def test_the_method_passes_its_switch_to_the_sampler(*, monkeypatch):
    seen = []
    original = LearnedSkillSampler.sample

    def sample(self, **kwargs):  # noqa: PLR0917 (bound method replacement)
        seen.append(kwargs.get("break_ties_among_best"))
        return original(self, **kwargs)

    monkeypatch.setattr(LearnedSkillSampler, "sample", sample)
    for switch in (False, True):
        method = _method(sampler_break_ties_among_best=switch)
        env = method.env
        state = env.build_initial_state(light_level=0.2, light_target=0.8)
        ground = GroundSkill(
            skill=LightSwitchSkills.TURN_ON_LIGHT,
            objects=(env.robot, env.get_cells()[-1], env.light),
        )
        method.execute_ground_skill(ground_skill=ground, state=state, explore=False)
    # Passed only when on, so a run with the switch off calls `sample` as it always did.
    assert seen == [None, True]
    assert EesMethod.model_fields["sampler_break_ties_among_best"].default is False


def test_a_snapshot_carries_the_switch_only_when_it_is_on():
    """Evaluation runs from the snapshot, so the switch must reach it; a run with the
    switch off writes the snapshot it always wrote."""
    off = DeploymentSnapshot.encode(method=_method())
    on = DeploymentSnapshot.encode(method=_method(sampler_break_ties_among_best=True))
    assert "sampler_break_ties_among_best" not in pickle.loads(off)["configuration"]
    assert pickle.loads(on)["configuration"]["sampler_break_ties_among_best"] is True
    env = LightSwitchEnvironment(grid_size=3)
    restored = DeploymentSnapshot.restore(
        raw=on, env=env, provider=LightSwitchSkillProvider(env=env)
    )
    assert restored.sampler_break_ties_among_best is True


def test_the_flag_is_off_unless_given():
    parser = argparse.ArgumentParser()
    EesCli.add_arguments(parser=parser)
    assert parser.parse_args([]).sampler_break_ties_among_best is False
    assert (
        parser.parse_args(["--sampler-break-ties-among-best"]).sampler_break_ties_among_best is True
    )
