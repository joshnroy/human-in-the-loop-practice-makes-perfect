"""Toss identity is independent of placement; its successful effects are not."""

import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    BIN_AT_SIDE,
    BIN_ON_GROUND,
    CUBE_AT_SIDE,
    GRASP_CLEAR,
    HAND_EMPTY,
    HOLDING,
    IN_BIN,
    PICK_PLANNABLE,
    ROBOT_AT_SIDE,
)
from hitl_pmp.environments.tossing3d.sides import Tossing3DSides
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.methods.belief_space.tossing3d_transition_model import apply_success_effects
from hitl_pmp.methods.practice_makes_perfect.dp_mdp import DpTaskModel
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.planning.fast_downward import FastDownwardPlanner
from hitl_pmp.planning.grounding import SkillGrounder


def _scene(*, bin_side, cube_side):
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    toss = GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        objects=(env.robot, env.bin, env.cube, env.barrier),
    )
    before = frozenset({
        GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube)),
        GroundAtom(predicate=BIN_ON_GROUND, objects=(env.bin,)),
        GroundAtom(predicate=BIN_AT_SIDE, objects=(env.bin, env.barrier, bin_side)),
        GroundAtom(predicate=CUBE_AT_SIDE, objects=(env.cube, env.barrier, cube_side)),
        GroundAtom(predicate=ROBOT_AT_SIDE, objects=(env.robot, env.barrier, Tossing3DSides.robot)),
        GroundAtom(predicate=GRASP_CLEAR, objects=(env.cube, env.bin)),
        GroundAtom(predicate=PICK_PLANNABLE, objects=(env.robot, env.cube)),
    })
    return env, provider, toss, before


def test_toss_has_one_grounding_for_both_bin_placements():
    candidates = []
    for bin_side in Tossing3DSides.objects():
        _, provider, toss, before = _scene(bin_side=bin_side, cube_side=Tossing3DSides.robot)
        applicable = SkillGrounder.applicable_ground_skills(
            skills=(toss.skill,), objects=provider.objects(), true_atoms=before
        )
        assert applicable == [toss]
        candidates.extend(applicable)
    assert len(set(candidates)) == 1
    assert tuple(parameter.name for parameter in candidates[0].skill.parameters) == (
        "robot",
        "bin",
        "cube",
        "barrier",
    )


@pytest.mark.parametrize("bin_side", Tossing3DSides.objects())
@pytest.mark.parametrize("cube_side", Tossing3DSides.objects())
def test_expectimax_success_replaces_old_side_and_preserves_pick_restriction(
    *, bin_side, cube_side
):
    env, provider, toss, before = _scene(bin_side=bin_side, cube_side=cube_side)
    after = apply_success_effects(
        true_atoms=before,
        ground_skill=toss,
        effects={toss: (toss.add_effects, toss.delete_effects, toss.ignore_effects)},
    )
    expected_side = GroundAtom(predicate=CUBE_AT_SIDE, objects=(env.cube, env.barrier, bin_side))
    assert {atom for atom in after if atom.predicate == CUBE_AT_SIDE} == {expected_side}
    assert GroundAtom(predicate=IN_BIN, objects=(env.cube, env.bin)) in after
    assert GroundAtom(predicate=HOLDING, objects=(env.robot, env.cube)) not in after
    assert {atom for atom in after if atom.predicate == BIN_AT_SIDE} == {
        atom for atom in before if atom.predicate == BIN_AT_SIDE
    }
    applicable_picks = SkillGrounder.applicable_ground_skills(
        skills=(Tossing3DSkills.PICK_CUBE,), objects=provider.objects(), true_atoms=after
    )
    assert bool(applicable_picks) == (bin_side == Tossing3DSides.robot)
    without_reachability = frozenset(atom for atom in after if atom.predicate != PICK_PLANNABLE)
    assert not SkillGrounder.applicable_ground_skills(
        skills=(Tossing3DSkills.PICK_CUBE,),
        objects=provider.objects(),
        true_atoms=without_reachability,
    )


@pytest.mark.parametrize("bin_side", Tossing3DSides.objects())
def test_symbolic_planner_uses_same_toss_for_either_landing_side(*, bin_side):
    other_side = next(side for side in Tossing3DSides.objects() if side != bin_side)
    env, provider, toss, before = _scene(bin_side=bin_side, cube_side=other_side)
    goal = frozenset({
        GroundAtom(predicate=IN_BIN, objects=(env.cube, env.bin)),
        GroundAtom(predicate=CUBE_AT_SIDE, objects=(env.cube, env.barrier, bin_side)),
    })
    plan = FastDownwardPlanner.plan(
        skills=(toss.skill,),
        predicates=provider.predicates(),
        types=provider.types(),
        objects=provider.objects(),
        init_atoms=before,
        goal=goal,
    )
    assert plan == [toss]


@pytest.mark.parametrize("bin_side", Tossing3DSides.objects())
def test_dp_success_copies_side_without_creating_distinct_toss_targets(*, bin_side):
    other_side = next(side for side in Tossing3DSides.objects() if side != bin_side)
    env, provider, toss, before = _scene(bin_side=bin_side, cube_side=other_side)
    goal = frozenset({
        GroundAtom(predicate=IN_BIN, objects=(env.cube, env.bin)),
        GroundAtom(predicate=CUBE_AT_SIDE, objects=(env.cube, env.barrier, bin_side)),
    })
    mdp = DpTaskModel.build(
        tasks=[(before, goal)], skills=(toss.skill,), objects=provider.objects(), max_states=10
    )
    assert {edge.skill for edge in mdp.transitions} == {toss}
    assert mdp.solve(competences={toss: 0.7})[0] == pytest.approx(0.7)
    contradictory_goal = goal | {
        GroundAtom(predicate=CUBE_AT_SIDE, objects=(env.cube, env.barrier, other_side))
    }
    impossible = DpTaskModel.build(
        tasks=[(before, contradictory_goal)],
        skills=(toss.skill,),
        objects=provider.objects(),
        max_states=10,
    )
    assert impossible.solve(competences={toss: 1.0})[0] == 0


@pytest.mark.parametrize("bin_side", Tossing3DSides.objects())
def test_ees_toss_setup_does_not_require_a_bin_relocation(*, bin_side):
    env, provider, toss, before = _scene(bin_side=bin_side, cube_side=Tossing3DSides.robot)
    method = EesMethod(env=env, skill_provider=provider, seed=0)
    assert (
        method.plan_to(init_atoms=before, goal=toss.preconditions, costs={}, practicing=True) == []
    )


@pytest.mark.parametrize("sharing", ["grounded", "parent"])
def test_toss_evidence_and_exploration_counts_follow_one_identity(*, sharing):
    env, provider, toss, _ = _scene(bin_side=Tossing3DSides.robot, cube_side=Tossing3DSides.robot)
    method = EesMethod(env=env, skill_provider=provider, seed=0, competence_sharing=sharing)
    for bin_side, success in zip(Tossing3DSides.objects(), [True, False], strict=True):
        _, _, target, before = _scene(bin_side=bin_side, cube_side=Tossing3DSides.robot)
        assert target.preconditions <= before
        method.observe_outcome(ground_skill=target, success=success, was_random_exploration=False)
        assert method.forecast_competence_model(ground_skill=target) is (
            method.forecast_competence_model(ground_skill=toss)
        )
    assert method.practice_target_num_tries(ground_skill=toss) == 2
    assert method.practice_candidates() == [toss]


def test_reset_destinations_still_control_the_same_toss_continuation():
    env, provider, toss, held = _scene(
        bin_side=Tossing3DSides.opposite, cube_side=Tossing3DSides.opposite
    )
    before = frozenset(atom for atom in held if atom.predicate != HOLDING) | {
        GroundAtom(predicate=HAND_EMPTY, objects=(env.robot,))
    }
    continuations = {}
    resets = provider.human_cube_bin_reset_skills()
    assert {reset.objects[-1] for reset in resets} == set(Tossing3DSides.objects())
    for reset in resets:
        atoms = apply_success_effects(
            true_atoms=before,
            ground_skill=reset,
            effects={reset: (reset.add_effects, reset.delete_effects, reset.ignore_effects)},
        )
        picks = SkillGrounder.applicable_ground_skills(
            skills=(Tossing3DSkills.PICK_CUBE,), objects=provider.objects(), true_atoms=atoms
        )
        assert len(picks) == 1
        pick = picks[0]
        atoms = apply_success_effects(
            true_atoms=atoms,
            ground_skill=pick,
            effects={pick: (pick.add_effects, pick.delete_effects, pick.ignore_effects)},
        )
        assert SkillGrounder.applicable_ground_skills(
            skills=(toss.skill,), objects=provider.objects(), true_atoms=atoms
        ) == [toss]
        after = apply_success_effects(
            true_atoms=atoms,
            ground_skill=toss,
            effects={toss: (toss.add_effects, toss.delete_effects, toss.ignore_effects)},
        )
        continuations[reset.objects[-1]] = bool(
            SkillGrounder.applicable_ground_skills(
                skills=(Tossing3DSkills.PICK_CUBE,), objects=provider.objects(), true_atoms=after
            )
        )
    assert continuations == {Tossing3DSides.robot: True, Tossing3DSides.opposite: False}
