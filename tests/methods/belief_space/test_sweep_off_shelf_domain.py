"""The SweepOffShelf3D practice domain: deployment value and belief routing."""

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects as O
from hitl_pmp.environments.sweepoffshelf3d.skills import SweepOffShelf3DSkills as S
from hitl_pmp.methods.belief_space.practice_domain import (
    RETURN_CUBE_BELIEF,
    SHELF_DEPLOYMENT_SKILLS,
    SHELF_ROBOT_SKILLS,
    SWEEP_OFF_SHELF_DOMAIN,
    SweepOffShelfDeployment,
)
from hitl_pmp.methods.belief_space.tossing3d_observation_model import skill_belief_model
from hitl_pmp.methods.belief_space.types.practice_example_source import PracticeExampleSource


def _value(*, horizon=4, **competence):
    competences = {name: np.array([competence.get(name, 0.0)]) for name in SHELF_ROBOT_SKILLS}
    return float(SweepOffShelfDeployment.values(competences=competences, horizon=horizon)[0])


def test_a_perfect_sweep_solves_in_two_skills():
    assert _value(horizon=2, PickWiper=1.0, SweepShelf=1.0) == 1.0
    assert _value(horizon=1, PickWiper=1.0, SweepShelf=1.0) == 0.0


def test_perfect_pick_and_place_needs_ten_skills():
    assert _value(horizon=10, PickCubeFromShelf=1.0, PlaceCubeInBox=1.0) == 1.0
    assert _value(horizon=9, PickCubeFromShelf=1.0, PlaceCubeInBox=1.0) == 0.0


def test_the_policy_takes_the_better_route():
    sweep_only = _value(horizon=12, PickWiper=0.9, SweepShelf=0.5)
    carry_only = _value(horizon=12, PickCubeFromShelf=0.95, PlaceCubeInBox=0.95)
    both = _value(
        horizon=12, PickWiper=0.9, SweepShelf=0.5, PickCubeFromShelf=0.95, PlaceCubeInBox=0.95
    )
    assert both >= max(sweep_only, carry_only) - 1e-12


def test_retries_raise_the_value_of_an_unreliable_sweep():
    assert _value(horizon=4, PickWiper=1.0, SweepShelf=0.5) == pytest.approx(1 - 0.5**3)


def test_every_shelf_skill_updates_its_own_belief():
    sweep = GroundSkill(skill=S.SWEEP_SHELF, objects=(O.robot, O.wiper, O.box, *O.cubes))
    model = skill_belief_model(ground_skill=sweep, domain=SWEEP_OFF_SHELF_DOMAIN)
    assert model.belief_name == "SweepShelf"
    assert model.example_source == PracticeExampleSource.SAMPLER
    reset = GroundSkill(skill=S.RESET_CUBES, objects=(O.robot, O.box, *O.cubes))
    model = skill_belief_model(ground_skill=reset, domain=SWEEP_OFF_SHELF_DOMAIN)
    assert model.belief_name == "reset_cubes"
    assert model.example_source == PracticeExampleSource.OUTCOME


def test_human_skills_are_inferred_not_assumed():
    """The robot is not told p: reset_cubes's competence is learned, never fixed at 1."""
    assert not SWEEP_OFF_SHELF_DOMAIN.deterministic_skills
    assert {"reset_cubes", "reset_wiper"} <= SWEEP_OFF_SHELF_DOMAIN.failure_effect_skills


def test_the_deployment_horizon_is_the_evaluation_episode_horizon():
    from hitl_pmp.environments.sweepoffshelf3d.problem import EPISODE_HORIZON

    assert SWEEP_OFF_SHELF_DOMAIN.deployment_horizon == EPISODE_HORIZON


def test_an_untrained_shelf_skill_is_believed_to_work_one_time_in_four():
    from hitl_pmp.methods.belief_space.competence_inference import InferenceConfig
    from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
        make_default_tossing3d_belief,
        refit_belief_state,
    )

    belief = refit_belief_state(
        state=make_default_tossing3d_belief(
            num_particles=4000,
            model="local_trend",
            engine="particle",
            inference_config=InferenceConfig(),
            domain=SWEEP_OFF_SHELF_DOMAIN,
        )
    )
    for name in ("SweepShelf", "PickCubeFromShelf", "reset_cubes"):
        assert belief.skill_beliefs[name].mean_competence() == pytest.approx(0.25, abs=0.02)
    assert 0.4 < SWEEP_OFF_SHELF_DOMAIN.expected_policy_value(beliefs=belief.skill_beliefs) < 0.7


def _shelf_atoms(*, on_shelf, in_box, wiper_home=True):
    from hitl_pmp.core.problem.tasks.types import GroundAtom
    from hitl_pmp.environments.sweepoffshelf3d.pick_plannable import PickPlannableAtoms
    from hitl_pmp.environments.sweepoffshelf3d.predicates import (
        SweepOffShelf3DAtoms,
        SweepOffShelf3DPredicates,
    )

    wiper = "WiperHome" if wiper_home else "WiperOnFloor"
    named = {("HandEmpty", ("robot",)), (wiper, ("wiper_0",))}
    named |= {("OnShelf", (cube.name,)) for cube in on_shelf}
    named |= {("InBox", (cube.name, O.box.name)) for cube in in_box}
    named = PickPlannableAtoms.with_verdicts(
        atoms=SweepOffShelf3DAtoms.with_summaries(atoms=frozenset(named)),
        plannable=lambda cube: True,
    )
    by_name = {p.name: p for p in SweepOffShelf3DPredicates.all()}
    objects = {obj.name: obj for obj in O.all_objects()}
    return (
        frozenset(
            GroundAtom(predicate=by_name[name], objects=tuple(objects[n] for n in names))
            for name, names in named
        )
        | SweepOffShelf3DPredicates.static_atoms()
    )


@pytest.mark.parametrize(
    ("ground", "atoms"),
    [
        (
            GroundSkill(skill=S.RESET_CUBES, objects=(O.robot, O.box, *O.cubes)),
            {"on_shelf": (), "in_box": O.cubes},
        ),
        (
            GroundSkill(skill=S.RESET_WIPER, objects=(O.robot, O.wiper)),
            {"on_shelf": O.cubes, "in_box": (), "wiper_home": False},
        ),
    ],
)
def test_a_human_skill_forecast_branches_on_its_inferred_competence(*, ground, atoms):
    """As for Tossing3D's human reset (#411): success with E[kappa], failure with
    1 - E[kappa], each branch conditioning only the human skill's own belief. A
    failed human skill moves nothing."""
    from hitl_pmp.methods.belief_space.competence_inference import InferenceConfig
    from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
        make_default_tossing3d_belief,
        mean_competence,
    )
    from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
        make_tossing3d_search_state,
        transition_outcomes,
    )

    priced = ground.model_copy(
        update={"skill": ground.skill.model_copy(update={"practice_cost": 2.0})}
    )
    state = make_default_tossing3d_belief(
        num_particles=64,
        model="local_trend",
        engine="grid",
        inference_config=InferenceConfig(),
        domain=SWEEP_OFF_SHELF_DOMAIN,
    )
    name = ground.skill.name
    assert name not in SWEEP_OFF_SHELF_DOMAIN.deterministic_skills
    before = _shelf_atoms(**atoms)
    competence = mean_competence(belief=state.skill_beliefs[name])
    assert competence == pytest.approx(0.25, abs=0.02)
    outcomes = transition_outcomes(
        environment_state=make_tossing3d_search_state(state=state, true_atoms=before),
        state=state,
        action=priced,
        ground_skills=(priced,),
        effects={priced: (priced.add_effects, priced.delete_effects, priced.ignore_effects)},
        exploration_epsilon=0.0,
        random_toss_competence=0.0,
        domain=SWEEP_OFF_SHELF_DOMAIN,
    )
    assert [p for p, _, _ in outcomes] == pytest.approx([competence, 1 - competence])
    (_, succeeded, after_success), (_, failed, after_failure) = outcomes
    assert priced.expected_add_effects(before_atoms=before) <= after_success
    assert after_failure == before
    assert mean_competence(belief=succeeded.skill_beliefs[name]) > competence
    assert mean_competence(belief=failed.skill_beliefs[name]) < competence
    for other in SHELF_ROBOT_SKILLS:
        assert succeeded.skill_beliefs[other] == state.skill_beliefs[other]


def test_both_return_skills_share_one_learned_belief():
    """One controller behind two operators (from the box, from the floor): one skill,
    so one competence belief, learned like every other from its outcomes."""
    for skill, cube_objects in (
        (S.RETURN_CUBE_TO_SHELF_FROM_BOX, (O.cubes[0], O.box)),
        (S.RETURN_CUBE_TO_SHELF_FROM_FLOOR, (O.cubes[0],)),
    ):
        ground = GroundSkill(skill=skill, objects=(O.robot, *cube_objects))
        model = skill_belief_model(ground_skill=ground, domain=SWEEP_OFF_SHELF_DOMAIN)
        assert model.belief_name == RETURN_CUBE_BELIEF
        assert model.example_source == PracticeExampleSource.OUTCOME
        assert skill.name in SWEEP_OFF_SHELF_DOMAIN.failure_effect_skills
    assert RETURN_CUBE_BELIEF in SHELF_ROBOT_SKILLS
    assert RETURN_CUBE_BELIEF not in SHELF_DEPLOYMENT_SKILLS


def test_an_untrained_return_is_believed_to_work_one_time_in_four():
    from hitl_pmp.methods.belief_space.competence_inference import InferenceConfig
    from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
        make_default_tossing3d_belief,
        mean_competence,
    )

    belief = make_default_tossing3d_belief(
        num_particles=64,
        model="local_trend",
        engine="grid",
        inference_config=InferenceConfig(),
        domain=SWEEP_OFF_SHELF_DOMAIN,
    )
    competence = mean_competence(belief=belief.skill_beliefs[RETURN_CUBE_BELIEF])
    assert competence == pytest.approx(0.25, abs=0.02)


def test_deployment_value_never_reads_the_return_belief():
    """Deployment solves a held-out task with no human, from every cube on the shelf;
    putting a cube back on the shelf never moves it toward the box, so J must not
    depend on how well the robot does it."""
    from hitl_pmp.methods.belief_space.competence_inference import InferenceConfig
    from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
        condition_skill_belief,
        make_default_tossing3d_belief,
    )

    belief = make_default_tossing3d_belief(
        num_particles=64,
        model="local_trend",
        engine="grid",
        inference_config=InferenceConfig(),
        domain=SWEEP_OFF_SHELF_DOMAIN,
    )
    before = SWEEP_OFF_SHELF_DOMAIN.expected_policy_value(beliefs=belief.skill_beliefs)
    trained = dict(belief.skill_beliefs)
    for _ in range(10):
        trained[RETURN_CUBE_BELIEF] = condition_skill_belief(
            belief=trained[RETURN_CUBE_BELIEF], success=True
        )
    assert SWEEP_OFF_SHELF_DOMAIN.expected_policy_value(beliefs=trained) == before


@pytest.mark.parametrize("solver", ["determinized_astar", "expectimax"])
def test_the_shelf_method_builds_from_the_shelfs_own_skills(*, solver: str) -> None:
    """Every belief the domain seeds is reachable from an offered skill, including a
    belief two operators share (ReturnCubeToShelf's FromBox and FromFloor). Built from
    the symbolic layer only; nothing here touches the simulator."""
    from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
    from hitl_pmp.environments.sweepoffshelf3d.skill_provider import (
        SweepOffShelf3DSkillProvider,
    )
    from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod

    env = SweepOffShelf3DEnvironment()
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=SweepOffShelf3DSkillProvider(env=env),
        pomdp_domain=SWEEP_OFF_SHELF_DOMAIN,
        pomdp_solver=solver,
        pomdp_search_depth=2,
        seed=0,
    )
    beliefs = set(method.pomdp_state.skill_beliefs)
    assert {name for name, _ in SWEEP_OFF_SHELF_DOMAIN.belief_sources} <= beliefs


@pytest.mark.parametrize("solver", ["determinized_astar", "expectimax"])
@pytest.mark.parametrize("human_reset", [True, False])
def test_the_shelf_method_builds_with_and_without_the_human_reset(
    *, solver: str, human_reset: bool
) -> None:
    """`--no-human-reset` offers neither human skill, so the practice domain drops
    their beliefs, aliases and failure-effect entries -- the same filtering the
    always-ask wrapper applies -- instead of failing its completeness check. With the
    reset offered, both human skills keep their beliefs and actions."""
    from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
    from hitl_pmp.environments.sweepoffshelf3d.skill_provider import (
        SweepOffShelf3DSkillProvider,
    )
    from hitl_pmp.methods.belief_space.practice_domain import SHELF_HUMAN_SKILLS
    from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod

    env = SweepOffShelf3DEnvironment()
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=SweepOffShelf3DSkillProvider(env=env, offer_human_reset=human_reset),
        pomdp_domain=SWEEP_OFF_SHELF_DOMAIN,
        pomdp_solver=solver,
        pomdp_search_depth=2,
        seed=0,
    )
    human = set(SHELF_HUMAN_SKILLS)
    domain = method.pomdp_domain
    beliefs = {name for name, _ in domain.belief_sources}
    actions = {ground.skill.name for ground in method._pomdp_model.ground_skills}  # noqa: SLF001
    assert (human <= beliefs) is human_reset
    assert (human <= set(method.pomdp_state.skill_beliefs)) is human_reset
    assert (human <= actions) is human_reset
    assert (human <= domain.failure_effect_skills) is human_reset
    if not human_reset:
        assert not (human & beliefs)
        assert not (human & set(method.pomdp_state.skill_beliefs))
        assert not (human & actions)
        assert not (human & domain.failure_effect_skills)
        assert not (human & {n for pair in domain.shared_beliefs for n in pair})
    # Robot beliefs are never dropped, and the shared domain is never modified.
    assert set(SHELF_ROBOT_SKILLS) <= beliefs
    assert human <= {name for name, _ in SWEEP_OFF_SHELF_DOMAIN.belief_sources}
