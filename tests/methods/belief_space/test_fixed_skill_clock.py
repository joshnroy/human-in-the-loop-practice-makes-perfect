"""Fixed-controller skills count every attempt as a training example, as the notebook does."""

from typing import Any

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.types import Tossing3DState
from hitl_pmp.methods.belief_space.competence_inference import BayesianSkillBelief
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    RESET_SKILL,
)
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state

from .test_notebook_equivalence import SKILLS, _notebook_run

FIXED = (PICK_SKILL, OPEN_GRIPPER_SKILL)
RESETS = (RESET_SKILL,)


def _method() -> Tossing3DPomdpMethod:
    env = Tossing3DEnvironment(scene_bg=False)
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        seed=3,
        pomdp_competence_model="global_curve",
        pomdp_inference_engine="grid",
        pomdp_num_particles=64,
        sampler_max_train_iters=1,
    )
    env.current_state = Tossing3DState(
        data={obj: np.zeros(obj.type.dim) for obj in method.objects()},
        abstract_atoms=frozenset(),
    )
    return method


def _ground(*, method: Tossing3DPomdpMethod, name: str) -> Any:
    return next(g for g in method._pomdp_model.ground_skills if g.skill.name == name)  # noqa: SLF001


def _belief(*, method: Tossing3DPomdpMethod, name: str) -> BayesianSkillBelief:
    belief = method.pomdp_state.skill_beliefs[name]
    assert isinstance(belief, BayesianSkillBelief)
    return belief


def _reset(*, method: Tossing3DPomdpMethod, name: str) -> None:
    method.record_action_cost(ground_skill=_ground(method=method, name=name))
    method.observe_help_granted(state=method.env.get_current_state())


def test_fixed_controller_and_reset_clocks_count_every_attempt() -> None:
    method = _method()
    attempts = dict.fromkeys((*FIXED, *RESETS), 0)
    for cycle, (n_fixed, n_resets) in enumerate(((3, 1), (0, 0), (2, 2))):
        for index in range(n_fixed):
            for name in FIXED:
                method.observe_outcome(
                    ground_skill=_ground(method=method, name=name), success=index % 2 == 0
                )
                attempts[name] += 1
        for _ in range(n_resets):
            for name in RESETS:
                _reset(method=method, name=name)
                attempts[name] += 1
        method.end_cycle()
        for name, count in attempts.items():
            belief = _belief(method=method, name=name)
            assert belief.total_training_examples == count, (cycle, name)


@pytest.mark.parametrize("skill", SKILLS)
def test_fixed_controller_grid_belief_matches_the_notebook(
    *, notebook: dict[str, Any], skill: str
) -> None:
    cycles = notebook["DATA"][skill][0]
    expected = _notebook_run(notebook=notebook, model="global_curve", cycles=cycles, patched=True)
    method = _method()
    pick = _ground(method=method, name=PICK_SKILL)
    filtered = []
    for m, n, s in cycles:
        assert _belief(method=method, name=PICK_SKILL).total_training_examples == m
        for index in range(n):
            method.observe_outcome(ground_skill=pick, success=index < s)
        filtered.append(_belief(method=method, name=PICK_SKILL).mean_competence())
        method.end_cycle()
    np.testing.assert_allclose(filtered, expected["filtered"], rtol=0, atol=1e-9)


@pytest.mark.parametrize("name", [*FIXED, RESET_SKILL])
def test_search_advances_the_same_clock_it_will_observe(*, name: str) -> None:
    method = _method()
    model = method._pomdp_model  # noqa: SLF001
    ground = _ground(method=method, name=name)
    state = method.pomdp_state
    outcomes = model.outcomes(
        environment_state=make_tossing3d_search_state(state=state, true_atoms=ground.preconditions),
        state=state,
        action=ground,
    )
    assert outcomes
    for _, branch, _ in outcomes:
        assert branch.pending_examples.get(name, 0) == 1
