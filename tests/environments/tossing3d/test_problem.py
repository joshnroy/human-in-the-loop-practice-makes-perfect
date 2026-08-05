"""Offline tests for `Tossing3DProblem` and `Tossing3DSkillProvider`.

`run_task_episode` itself needs a simulator and lives in `test_kinder_fidelity.py`.
"""

import numpy as np

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import (
    HAND_EMPTY,
    HOLDING,
    IN_GOAL_REGION,
    NEAR_BIN,
    ON_GROUND,
    REACHABLE,
)
from hitl_pmp.environments.tossing3d.problem import Tossing3DProblem
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks
from hitl_pmp.planning.grounding import SkillGrounder

from .observations import COINCIDENT_BIN_X, state


def _problem() -> Tossing3DProblem:
    env = Tossing3DEnvironment()
    return Tossing3DProblem(env=env, tasks=Tossing3DTasks(env=env, seed=0))


def test_the_horizon_is_the_shortest_solve_plus_two() -> None:
    """Three skills is the shortest solve and there is no shorter route: `Toss` requires
    both `Holding` and `NearBin`, and nothing else grants either. The `+ 2` buys one
    recovery from a failed grasp -- and no more, because after a toss there is nothing to
    recover from anyway."""
    assert _problem().max_episode_steps() == 5


def test_the_problem_has_no_human_oracle_yet() -> None:
    """Stated as a test rather than left implicit, because this is the domain in the repo
    that most wants one: a tossed cube genuinely needs someone to walk over and pick it
    up, and `Metrics.num_human_interventions()` reporting (0.0, 0) is a gap in what is
    representable, not evidence that no intervention was needed."""
    assert _problem().human is None


def test_the_provider_exposes_every_skill_predicate_type_and_object() -> None:
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment())
    assert provider.skills() == (
        Tossing3DSkills.PICK,
        Tossing3DSkills.MOVE_TO_THROW_POSE,
        Tossing3DSkills.TOSS,
    )
    assert set(provider.predicates()) == {
        IN_GOAL_REGION,
        HAND_EMPTY,
        HOLDING,
        ON_GROUND,
        REACHABLE,
        NEAR_BIN,
    }
    assert {obj.type for obj in provider.objects()} == set(provider.types())


def test_every_predicate_a_skill_references_is_one_the_provider_publishes() -> None:
    """`SkillGrounder` abstracts a state using `provider.predicates()` and checks
    applicability against `Skill.preconditions`. A precondition over a predicate the
    provider does not publish is never true, so the skill silently becomes unusable."""
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment())
    published = set(provider.predicates())
    for skill in provider.skills():
        for atom in (*skill.preconditions, *skill.add_effects, *skill.delete_effects):
            assert atom.predicate in published, f"{skill.name} references {atom.predicate.name}"


def test_the_symbolic_layer_grounds_the_oracles_own_plan_shape() -> None:
    """Walked with the real grounder rather than by hand: from the initial abstract state
    only `Pick` is applicable; holding the cube unlocks `MoveToThrowPose`; standing near
    the bin while holding unlocks `Toss`."""
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment())

    def applicable(**kwargs) -> set[str]:
        current = state(**kwargs)
        atoms = SkillGrounder.abstract_state(
            state=current, objects=provider.objects(), predicates=provider.predicates()
        )
        return {
            ground.skill.name
            for ground in SkillGrounder.applicable_ground_skills(
                skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
            )
        }

    assert applicable() == {"Pick"}
    assert applicable(gripper=0.9, cube_z=0.4) == {"MoveToThrowPose"}
    assert applicable(gripper=0.9, cube_z=0.4, base_x=COINCIDENT_BIN_X - 1.35) == {
        "MoveToThrowPose",
        "Toss",
    }


def test_nothing_is_applicable_once_the_cube_is_past_the_barrier() -> None:
    """The irreversibility, read off the symbolic layer: after a toss the cube is beyond
    the barrier, `Reachable` is false, `Pick` is inapplicable, and no skill remains. A
    planner asked to recover from here correctly finds no plan."""
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment())
    landed = state(cube_x=2.6, cube_z=0.025, gripper=0.0, base_x=COINCIDENT_BIN_X - 1.35)
    atoms = SkillGrounder.abstract_state(
        state=landed, objects=provider.objects(), predicates=provider.predicates()
    )
    assert not SkillGrounder.applicable_ground_skills(
        skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
    )


def test_the_provider_delegates_sampling_and_encoding_to_the_skills_container() -> None:
    env = Tossing3DEnvironment()
    provider = Tossing3DSkillProvider(env=env)
    ground_skill = GroundSkill(
        skill=Tossing3DSkills.MOVE_TO_THROW_POSE, objects=(env.robot, env.cube, env.bin)
    )
    params = provider.sample_params(ground_skill=ground_skill, rng=np.random.default_rng(0))
    action = provider.compute_action(ground_skill=ground_skill, params=params, state=state())
    assert action[0] == Tossing3DEnvironment.move_to_throw_pose_id
