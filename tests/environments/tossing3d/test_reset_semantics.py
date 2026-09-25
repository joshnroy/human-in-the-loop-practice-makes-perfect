"""Reset forecasts preserve the gripper command while removing its grasp."""

import pytest

from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.environments.tossing3d.predicates import CLOSED_EMPTY, HAND_EMPTY, HOLDING, ON_GROUND
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.belief_space.tossing3d_transition_model import apply_success_effects


@pytest.mark.parametrize("layout", list(Tossing3DLayout))
@pytest.mark.parametrize("reset_index", [0, 1])
@pytest.mark.parametrize("gripper", ["open", "holding", "closed_empty"])
def test_reset_forecast_preserves_command_and_removes_grasp(*, layout, reset_index, gripper):
    env = Tossing3DEnvironment(layout=layout)
    reset = Tossing3DSkillProvider(env=env).movables_reset_skills()[reset_index]
    empty = GroundAtom(predicate=HAND_EMPTY, objects=(env.robot,))
    holding = GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube))
    closed = GroundAtom(predicate=CLOSED_EMPTY, objects=(env.robot, env.cube))
    before = reset.preconditions | frozenset({
        {"open": empty, "holding": holding, "closed_empty": closed}[gripper]
    })
    after = apply_success_effects(
        true_atoms=before,
        ground_skill=reset,
        effects={reset: (reset.add_effects, reset.delete_effects, reset.ignore_effects)},
    )
    assert holding not in after
    assert (empty in after) == (gripper == "open")
    assert (closed in after) == (gripper != "open")
    assert reset.preconditions <= before
    assert reset.evaluate_practice_cost() == 5.0


@pytest.mark.parametrize("layout", list(Tossing3DLayout))
@pytest.mark.parametrize("reset_index", [0, 1])
def test_classical_plan_opens_gripper_after_resetting_a_held_cube(*, layout, reset_index):
    from hitl_pmp.environments.tossing3d.recovery_skills import ON_FLOOR
    from hitl_pmp.planning.fast_downward import FastDownwardPlanner

    env = Tossing3DEnvironment(layout=layout)
    provider = Tossing3DSkillProvider(env=env)
    reset = provider.movables_reset_skills()[reset_index]
    opened = next(skill for skill in provider.skills() if skill.name == "OpenGripper")
    before = reset.preconditions | frozenset({
        GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube))
    })
    floor = GroundAtom(
        predicate=ON_FLOOR if layout == Tossing3DLayout.SAME_SIDE else ON_GROUND,
        objects=(env.cube, env.bin) if layout == Tossing3DLayout.SAME_SIDE else (env.cube,),
    )
    goal = frozenset({floor, GroundAtom(predicate=HAND_EMPTY, objects=(env.robot,))})
    plan = FastDownwardPlanner.plan(
        skills=(reset.skill, opened),
        predicates=provider.predicates(),
        types=provider.types(),
        objects=provider.objects(),
        init_atoms=before,
        goal=goal,
    )
    assert [step.skill.name for step in plan] == [reset.skill.name, "OpenGripper"]
    for step in plan:
        assert step.preconditions <= before
        before = apply_success_effects(
            true_atoms=before,
            ground_skill=step,
            effects={step: (step.add_effects, step.delete_effects, step.ignore_effects)},
        )
    assert goal <= before


def test_conditional_reset_planning_preserves_ground_costs_and_translation_cache() -> None:
    from hitl_pmp.planning.fast_downward import FastDownwardPlanner, PlanningFailure
    from hitl_pmp.planning.types import TranslationCache

    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    human = provider.human_cube_bin_reset_skill()
    automatic = provider.non_human_cube_bin_reset_skill()
    holding = GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube))
    closed = GroundAtom(predicate=CLOSED_EMPTY, objects=(env.robot, env.cube))
    empty = GroundAtom(predicate=HAND_EMPTY, objects=(env.robot,))
    cache = TranslationCache()
    options = {
        "skills": (human.skill, automatic.skill),
        "predicates": provider.predicates(),
        "types": provider.types(),
        "objects": provider.objects(),
        "goal": frozenset({closed, *human.add_effects}),
        "translation_cache": cache,
    }
    for preferred, other in ((human, automatic), (automatic, human)):
        plan = FastDownwardPlanner.plan(
            **options,
            init_atoms=human.preconditions | frozenset({holding}),
            ground_skill_costs={preferred: 0.5, other: 2.0},
        )
        assert plan == [preferred]
    with pytest.raises(PlanningFailure):
        FastDownwardPlanner.plan(**options, init_atoms=human.preconditions | frozenset({empty}))
