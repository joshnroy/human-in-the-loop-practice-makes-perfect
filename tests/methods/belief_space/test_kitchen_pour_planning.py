"""The planner cannot imagine a second kitchen pour before the jug is refilled.

Replaying eight finished runs of the 2026-11-11 gripper sweep, the Curtis planner's
chosen path held two or more pours in a row in 435/469 non-STOP decisions (up to 17),
because a forecast success left all three dumplings in the jug.
"""

import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.methods.belief_space.determinized import DeterminizedAStarPlanner
from hitl_pmp.methods.belief_space.failure_effect_model import EmpiricalFailureEffects
from hitl_pmp.methods.belief_space.skill_plans_domain import (
    POURING_KITCHEN_SKILL_STEPS,
    SkillPlansPracticeDomain,
)
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_observation_model import make_default_tossing3d_belief
from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.practice_example_source import PracticeExampleSource
from hitl_pmp.methods.belief_space.types.search_trace import SearchTrace
from tests.environments.pouring3d.kitchen_symbolic import KitchenSymbols as K

HUMAN = "human_bulk_pour_back"
POURS = ("Pour1", "Pour2", "Pour3")
SUPPORT = (
    "PickPitcher",
    "PlacePitcherAtHome",
    "PickShrimpFromTray",
    "PickShrimpFromFloor",
    "DropShrimpIntoPitcher",
    HUMAN,
    "PickShrimpFromSurface",
)


def _domain() -> SkillPlansPracticeDomain:
    """`Tossing3DPomdpCli`'s pouring3d_kitchen domain for requested counts 1, 2, 3."""
    return SkillPlansPracticeDomain(
        deployment_horizon=3,
        deployment_plans=tuple(("PickPitcher", pour, "PlacePitcherAtHome") for pour in POURS),
        belief_sources=(
            *(
                (
                    name,
                    PracticeExampleSource.NO_TRAINING
                    if name == HUMAN
                    else PracticeExampleSource.OUTCOME,
                )
                for name in SUPPORT
            ),
            *((name, PracticeExampleSource.SAMPLER) for name in POURS),
        ),
        initial_competence_alpha=1.0,
        initial_competence_beta=3.0,
        default_search_depth=4,
        human_skill_names=frozenset({HUMAN}),
        failure_effect_skills=frozenset((*SUPPORT, *POURS)),
        skill_steps=POURING_KITCHEN_SKILL_STEPS,
    )


def _spilled_everything(*, before: frozenset[GroundAtom], name: str) -> frozenset[GroundAtom]:
    """What 39/43 logged failed pours left: the whole payload in the skillet. A failed
    Pour3 cannot end there (that is its success), so one of its dumplings misses."""
    names = [d.name for d in K.DUMPLINGS]
    kept = frozenset(a for a in before if a.predicate.name != "InPitcher")
    if name == "Pour3":
        return kept | K.atoms(
            texts=[
                *(f"InTray({name},tray)" for name in names[:2]),
                f"OnFloor({names[2]})",
                *(f"Settled({name})" for name in names),
            ]
        )
    return kept | K.atoms(
        texts=[
            *(f"InTray({name},tray)" for name in names),
            *(f"Settled({name})" for name in names),
            "Split3()",
        ]
    )


def _setup() -> tuple[Tossing3DPracticeModel, Tossing3DBeliefState, frozenset[GroundAtom]]:
    domain = _domain()
    grounds = K.ground_skills()
    start = K.holding_a_full_jug()
    counts: tuple = ()
    for ground in grounds:
        if ground.skill.name in POURS:
            counts = EmpiricalFailureEffects.observe(
                counts=counts,
                ground_skill=ground,
                before_atoms=start,
                after_atoms=_spilled_everything(before=start, name=ground.skill.name),
                was_random_exploration=False,
            )
    model = Tossing3DPracticeModel(
        domain=domain,
        ground_skills=grounds,
        exploration_epsilon=0.0,
        linear_cost_lambda=0.0003,
        failure_effect_counts=counts,
    )
    state = make_default_tossing3d_belief(domain=domain, num_particles=64).model_copy(
        update={
            "expected_execution_costs": {
                str(g): 20.0 if g.skill.name == HUMAN else 5.0 for g in grounds
            }
        }
    )
    return model, state, start


def _names(*, path: list[dict]) -> list[str]:
    return [step["skill"]["name"] for step in path]


@pytest.mark.parametrize("variant", ["curtis", "algorithm3"])
def test_no_imagined_path_pours_twice_without_refilling_the_jug(*, variant: str) -> None:
    model, state, start = _setup()
    trace = SearchTrace()
    DeterminizedAStarPlanner(
        max_iterations=100,
        seed=0,
        observation_probability_weight=0.01,
        variant=variant,  # type: ignore[arg-type]
    ).solve(
        environment_state=make_tossing3d_search_state(state=state, true_atoms=start),
        summed_cost=state.accumulated_cost,
        belief_state=state,
        remaining_actions=20,
        model=model,
        num_samples=1,
        trace=trace,
    )
    paths = [
        _names(path=event["best_action_path"])
        for event in trace.events
        if event["event"] == "action_value" and "best_action_path" in event
    ]
    assert any(path[0] in POURS for path in paths), "the search never tried a pour"
    for path in paths:
        assert not any(
            first in POURS and second in POURS
            for first, second in zip(path, path[1:], strict=False)
        ), path


def test_after_any_forecast_pour_the_only_way_to_pour_again_is_to_refill() -> None:
    model, state, start = _setup()
    search = make_tossing3d_search_state(state=state, true_atoms=start)
    pours: list[GroundSkill] = [g for g in model.ground_skills if g.skill.name in POURS]
    assert set(pours) <= set(model.get_valid_actions(environment_state=search))
    for pour, subsets in zip(pours, (3, 3, 1), strict=True):
        children = model.transition_outcomes(
            environment_state=search, practice_action=pour, belief_state=state
        )
        assert sum(probability for _, _, probability in children) == pytest.approx(1.0)
        # One child per subset that can leave, and the one recorded failure effect.
        assert len({child.true_atoms for child, _, _ in children}) == subsets + 1
        for child, _, _ in children:
            assert not set(pours) & set(model.get_valid_actions(environment_state=child))
