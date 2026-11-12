"""The fitted real toss classifier ignores the standoff; the executed action keeps it.

These build a `LearnedSkillSampler`, so they import torch, and they live here rather than in
`tests/environments/tossing3d/` because CI runs that folder under MuJoCo's OSMesa, whose
LLVM and torch's bundled triton abort the worker when one process loads both.
"""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.environments.tossing3d.toss_direction import (
    TossDirectionChoice,
    TossDirectionSelector,
)
from hitl_pmp.environments.tossing3d.wide_long_range_proposal import (
    REAL_TOSS_PARAMETERS,
    WideLongRangeTossProposal,
)
from hitl_pmp.methods.practice_makes_perfect.wrapped_sampler import LearnedSkillSampler
from tests.environments.tossing3d.observations import state


def test_two_real_candidates_differing_only_in_standoff_score_identically() -> None:
    env = Tossing3DEnvironment(toss_parameters=REAL_TOSS_PARAMETERS)
    sampler = LearnedSkillSampler(
        skill_name="MoveToTossLocationAndToss", param_dim=3, seed=0, max_train_iters=200
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        params = WideLongRangeTossProposal.sample(rng=rng, parameters=REAL_TOSS_PARAMETERS)
        sampler.observe(sampler_input=_toss_row(env=env, params=params), success=params[1] < 120)
    sampler.fit()
    assert sampler.is_fitted
    rows = [_toss_row(env=env, params=np.array([s, 114.72, 729.37])) for s in (1.395, 1.405)]
    low, high = sampler.score_inputs(sampler_inputs=rows)
    assert low == high


def test_the_executed_real_action_still_carries_the_drawn_standoff(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    def select_for_state(*, state, standoff: float) -> TossDirectionChoice:
        del state, standoff
        return TossDirectionChoice(
            direction_deg=0, rotation=0.0, stand_xy=(0.0, 0.0), clearance_m=1.0
        )

    monkeypatch.setattr(TossDirectionSelector, "select_for_state", select_for_state)
    env = Tossing3DEnvironment(toss_parameters=REAL_TOSS_PARAMETERS)
    candidates = [np.array([1.196, 114.72, 729.37]), np.array([1.204, 114.72, 729.37])]
    rows = [_toss_row(env=env, params=candidate) for candidate in candidates]
    assert rows[0] == rows[1]
    sampler = LearnedSkillSampler(skill_name="MoveToTossLocationAndToss", param_dim=3, seed=0)
    chosen = sampler.sample(sampler_inputs=rows, candidates=candidates, explore=False).params
    assert any(chosen is candidate for candidate in candidates)
    action = Tossing3DSkillProvider(env=env).compute_action(
        ground_skill=_toss(env=env), params=chosen, state=state(env=env)
    )
    assert action[1] == pytest.approx(chosen[0])


def _toss_row(*, env: Tossing3DEnvironment, params: np.ndarray) -> list[float]:
    row = Tossing3DSkillProvider(env=env).hand_selected_feature_transform(
        ground_skill=_toss(env=env), state=state(env=env), params=params
    )
    assert row is not None
    return row


def _toss(*, env: Tossing3DEnvironment) -> GroundSkill:
    return GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier),
    )
