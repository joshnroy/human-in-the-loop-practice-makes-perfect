"""SweepTable belief routing and joint deployment success, without a simulator."""

import argparse

import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.sweeptable3d.objects import SweepTable3DObjects as O
from hitl_pmp.environments.sweeptable3d.predicates import (
    HOLDING_WIPER,
    IN_BOX,
    ON_TABLE,
    TASK_ACTIVE,
    SweepTable3DAtoms,
    SweepTable3DPredicates,
)
from hitl_pmp.environments.sweeptable3d.skill_provider import SweepTable3DSkillProvider
from hitl_pmp.environments.sweeptable3d.skills import SweepTable3DSkills as S
from hitl_pmp.methods.belief_space.competence_inference import InferenceConfig
from hitl_pmp.methods.belief_space.expectimax import ExpectimaxPlanner
from hitl_pmp.methods.belief_space.failure_effect_model import EmpiricalFailureEffects
from hitl_pmp.methods.belief_space.practice_domain import SWEEP_OFF_SHELF_DOMAIN
from hitl_pmp.methods.belief_space.sweeptable3d_domain import (
    RETURN_CUBE_BELIEF,
    SWEEP_TABLE_DOMAIN,
    TABLE_DEPLOYMENT_SKILLS,
    TABLE_ROBOT_SKILLS,
    SweepTableDeployment,
)
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    condition_skill_belief,
    make_default_tossing3d_belief,
    mean_competence,
    skill_belief_model,
)
from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
    GroundSkillText,
    make_tossing3d_search_state,
)
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.practice_example_source import PracticeExampleSource
from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis, WeightedHypothesis
from hitl_pmp.methods.belief_space.types.stop_action import STOP_ACTION
from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import WeightedHypothesisBelief
from hitl_pmp.methods.practice_makes_perfect.cli import Tossing3DPomdpCli
from hitl_pmp.planning.grounding import SkillGrounder
from tests.environments.sweeptable3d.test_skill_provider import SymbolicEnvironment


def _value(*, horizon=12, **competences):
    values = {name: np.array([competences.get(name, 0.0)]) for name in TABLE_DEPLOYMENT_SKILLS}
    return float(SweepTableDeployment.values(competences=values, horizon=horizon)[0])


def _belief():
    return make_default_tossing3d_belief(
        num_particles=32,
        model="local_trend",
        engine="grid",
        inference_config=InferenceConfig(),
        domain=SWEEP_TABLE_DOMAIN,
    )


def _atoms(*, on_table=O.cubes, in_box=(), on_floor=(), active=True, holding_wiper=True):
    named = {("OnTable", (cube.name,)) for cube in on_table}
    named |= {("InBox", (cube.name, O.box.name)) for cube in in_box}
    named |= {("OnFloor", (cube.name,)) for cube in on_floor}
    if active:
        named.add(("TaskActive", ()))
    if holding_wiper:
        named.add(("HoldingWiper", (O.robot.name, O.wiper.name)))
    else:
        named |= {("HandEmpty", (O.robot.name,)), ("WiperHome", (O.wiper.name,))}
    summarized = SweepTable3DAtoms.with_summaries(atoms=frozenset(named))
    predicates = {predicate.name: predicate for predicate in SweepTable3DPredicates.all()}
    objects = {obj.name: obj for obj in O.all_objects()}
    return SweepTable3DPredicates.static_atoms() | frozenset(
        GroundAtom(predicate=predicates[name], objects=tuple(objects[obj] for obj in names))
        for name, names in summarized
    )


def _sweep():
    return GroundSkill(skill=S.SWEEP_TABLE, objects=(O.robot, O.wiper, O.box, *O.cubes))


def test_cli_routes_table_to_its_own_domain_and_preserves_shelf():
    assert Tossing3DPomdpCli.DOMAINS["SweepTable3DCli"] is SWEEP_TABLE_DOMAIN
    assert Tossing3DPomdpCli.DOMAINS["SweepOffShelf3DCli"] is SWEEP_OFF_SHELF_DOMAIN


@pytest.mark.parametrize("depth", [None, 4])
def test_cli_accepts_table_and_resolves_search_depth_without_starting_environment(*, depth):
    seen = []

    class SweepTable3DCli:
        @staticmethod
        def run_method(*, args, **kwargs):
            del kwargs
            seen.append((args.pomdp_search_depth, args.defer_rendering))

    parser = argparse.ArgumentParser()
    Tossing3DPomdpCli.add_arguments(parser=parser)
    parser.set_defaults(seed=0, output_dir=None)
    args = parser.parse_args([] if depth is None else ["--pomdp-search-depth", str(depth)])
    Tossing3DPomdpCli.run(args=args, env_cli=SweepTable3DCli)
    assert seen == [(SWEEP_TABLE_DOMAIN.default_search_depth if depth is None else depth, True)]


def test_one_joint_sweep_solves_all_five_in_two_skills():
    assert _value(horizon=1, PickWiper=1.0, SweepTable=1.0) == 0.0
    assert _value(horizon=2, PickWiper=1.0, SweepTable=1.0) == 1.0
    assert _value(horizon=2, PickWiper=0.8, SweepTable=0.3) == pytest.approx(0.8 * 0.3)


def test_sweep_competence_is_one_joint_outcome_not_five_independent_outcomes():
    assert _value(horizon=2, PickWiper=1.0, SweepTable=0.4) == pytest.approx(0.4)
    assert _value(horizon=12, PickWiper=1.0, SweepTable=0.4) == pytest.approx(0.4)


def test_deployment_failures_do_not_gain_optimistic_retries():
    assert _value(horizon=4, PickWiper=0.5, SweepTable=0.6) == pytest.approx(0.5 * 0.6)
    assert _value(horizon=100, PickWiper=0.5, SweepTable=0.6) == pytest.approx(0.5 * 0.6)


def test_individual_table_delivery_cannot_bypass_the_joint_sweep():
    assert _value(horizon=9, PickCubeFromTable=1.0, PlaceCubeInBox=1.0) == 0.0
    assert _value(horizon=10, PickCubeFromTable=1.0, PlaceCubeInBox=1.0) == 0.0
    assert _value(
        horizon=10, PickWiper=1.0, SweepTable=0.4, PickCubeFromTable=1.0, PlaceCubeInBox=1.0
    ) == pytest.approx(0.4)


def test_vectorized_values_and_zero_horizon():
    competences = {name: np.zeros(3) for name in TABLE_DEPLOYMENT_SKILLS}
    competences["PickWiper"][:] = [1.0, 0.5, 0.0]
    competences["SweepTable"][:] = [0.3, 0.8, 1.0]
    np.testing.assert_allclose(
        SweepTableDeployment.values(competences=competences, horizon=2), [0.3, 0.4, 0.0]
    )
    np.testing.assert_array_equal(
        SweepTableDeployment.values(competences=competences, horizon=0), np.zeros(3)
    )


def test_deployment_uses_only_tool_pick_and_one_joint_sweep():
    assert TABLE_DEPLOYMENT_SKILLS == ("PickWiper", "SweepTable")
    assert {atom.predicate.name for atom in SweepTable3DPredicates.goal_atoms()} == {
        "InBox",
        "SweepSucceeded",
    }
    delivered_without_sweeping = _atoms(on_table=(), in_box=O.cubes)
    assert not SweepTable3DPredicates.goal_atoms() <= delivered_without_sweeping


def test_every_offered_skill_has_a_belief_and_only_sweep_uses_sampler_evidence():
    for skill in (*S.robot_skills(), *S.human_skills()):
        belief_name = SWEEP_TABLE_DOMAIN.belief_name(skill_name=skill.name)
        assert belief_name is not None
        source = SWEEP_TABLE_DOMAIN.example_source(belief_name=belief_name)
        expected = (
            PracticeExampleSource.SAMPLER
            if skill == S.SWEEP_TABLE
            else PracticeExampleSource.OUTCOME
        )
        assert source == expected
        assert skill.name in SWEEP_TABLE_DOMAIN.failure_effect_skills
    assert SWEEP_TABLE_DOMAIN.sampler_skill == "SweepTable"
    assert not SWEEP_TABLE_DOMAIN.deterministic_skills
    assert SWEEP_TABLE_DOMAIN.human_skill_names == {"reset_cubes", "reset_wiper"}


def test_return_operators_share_their_own_table_belief():
    for skill in S.RETURN_CUBE_SKILLS:
        assert SWEEP_TABLE_DOMAIN.belief_name(skill_name=skill.name) == RETURN_CUBE_BELIEF
    assert RETURN_CUBE_BELIEF == "ReturnCubeToTable"
    assert not any("Shelf" in name for name, _ in SWEEP_TABLE_DOMAIN.belief_sources)


def test_table_domain_reuses_prior_and_search_configuration_without_changing_shelf():
    assert SWEEP_TABLE_DOMAIN.initial_competence_alpha == 1.0
    assert SWEEP_TABLE_DOMAIN.initial_competence_beta == 3.0
    assert SWEEP_TABLE_DOMAIN.deployment_horizon == 12
    assert SWEEP_TABLE_DOMAIN.default_search_depth == 7
    assert SWEEP_OFF_SHELF_DOMAIN.sampler_skill == "SweepShelf"


def test_deployment_theta_is_sampled_only_from_joint_sweep_beliefs():
    beliefs = _belief().skill_beliefs
    seen = []

    def sample_skills(*, belief, count):
        seen.append(belief)
        return [SkillHypothesis(competence=0.5, learning_rate=0.0)] * count

    thetas = SWEEP_TABLE_DOMAIN.sample_thetas(
        beliefs=beliefs,
        sample_skills=lambda belief, count: sample_skills(belief=belief, count=count),
        count=3,
    )
    assert len(thetas) == 3
    assert all(set(theta) == set(TABLE_DEPLOYMENT_SKILLS) for theta in thetas)
    assert seen == [beliefs[name] for name in TABLE_DEPLOYMENT_SKILLS]
    expected = _value(**dict.fromkeys(TABLE_DEPLOYMENT_SKILLS, 0.5))
    assert SWEEP_TABLE_DOMAIN.evaluate_theta(theta=thetas[0]) == pytest.approx(expected)


def test_expected_deployment_value_is_repeatable_and_excludes_reset_beliefs():
    beliefs = _belief().skill_beliefs
    before = SWEEP_TABLE_DOMAIN.expected_policy_value(beliefs=beliefs)
    assert 0.0 < before < 1.0
    assert SWEEP_TABLE_DOMAIN.expected_policy_value(beliefs=beliefs) == before
    trained = dict(beliefs)
    for name in set(beliefs) - set(TABLE_DEPLOYMENT_SKILLS):
        trained[name] = condition_skill_belief(belief=trained[name], success=True)
    assert SWEEP_TABLE_DOMAIN.expected_policy_value(beliefs=trained) == before
    values = SWEEP_TABLE_DOMAIN.sample_policy_values(
        beliefs=beliefs, rng=np.random.default_rng(0), count=4
    )
    assert values.shape == (4,)
    assert ((values >= 0.0) & (values <= 1.0)).all()


def test_a_sweep_branches_on_one_binary_outcome_with_all_five_success_effects():
    belief = _belief()
    sweep = _sweep()
    model = Tossing3DPracticeModel(
        domain=SWEEP_TABLE_DOMAIN, ground_skills=(sweep,), exploration_epsilon=0.0
    )
    before = _atoms()
    outcomes = model.outcomes(
        environment_state=make_tossing3d_search_state(state=belief, true_atoms=before),
        state=belief,
        action=sweep,
    )
    assert len(outcomes) == 2
    success_probability = mean_competence(belief=belief.skill_beliefs["SweepTable"])
    assert [probability for probability, _, _ in outcomes] == pytest.approx([
        success_probability,
        1.0 - success_probability,
    ])
    _, succeeded, after = outcomes[0]
    assert SweepTable3DPredicates.goal_atoms() <= after
    assert not any(atom.predicate == ON_TABLE for atom in after)
    assert succeeded.sampler_training["SweepTable"].successes == 1
    assert outcomes[1][1].sampler_training["SweepTable"].failures == 1


def test_partial_sweep_failure_conditions_once_and_terminal_effects_block_forward_skills():
    belief = _belief()
    sweep = _sweep()
    reset = GroundSkill(skill=S.RESET_CUBES, objects=(O.robot, O.box, O.wiper, *O.cubes))
    before = _atoms()
    partial = _atoms(on_table=(), in_box=O.cubes[:4], on_floor=O.cubes[4:], active=False)
    counts = EmpiricalFailureEffects.observe(
        counts=(),
        ground_skill=sweep,
        before_atoms=before,
        after_atoms=partial,
        was_random_exploration=False,
    )
    model = Tossing3DPracticeModel(
        domain=SWEEP_TABLE_DOMAIN,
        ground_skills=(sweep, reset),
        exploration_epsilon=0.0,
        failure_effect_counts=counts,
    )
    outcomes = model.outcomes(
        environment_state=make_tossing3d_search_state(state=belief, true_atoms=before),
        state=belief,
        action=sweep,
    )
    _, failed, after = outcomes[1]
    assert after == partial
    assert failed.sampler_training["SweepTable"].failures == 1
    assert failed.sampler_training["SweepTable"].successes == 0
    assert mean_competence(belief=failed.skill_beliefs["SweepTable"]) < mean_competence(
        belief=belief.skill_beliefs["SweepTable"]
    )
    assert model.get_valid_actions(
        environment_state=make_tossing3d_search_state(state=failed, true_atoms=after)
    ) == [reset]


def test_nonterminal_partial_sweep_keeps_individual_delivery_available():
    before = _atoms(on_table=O.cubes[4:], in_box=O.cubes[:4], holding_wiper=False)
    available = SkillGrounder.applicable_ground_skills(
        skills=S.robot_skills(), objects=O.all_objects(), true_atoms=before
    )
    assert any(ground.skill == S.PICK_CUBE_FROM_TABLE for ground in available)
    assert all(ground.skill != S.SWEEP_TABLE for ground in available)


def test_ball_reset_charges_cost_and_requires_tool_return_before_rearming_practice():
    belief = _belief()
    reset = GroundSkill(
        skill=S.RESET_CUBES.model_copy(update={"practice_cost": 3.0}),
        objects=(O.robot, O.box, O.wiper, *O.cubes),
    )
    belief = belief.model_copy(
        update={"expected_execution_costs": {GroundSkillText.of(ground_skill=reset): 3.0}}
    )
    sweep = _sweep()
    return_tool = GroundSkill(skill=S.RETURN_WIPER, objects=(O.robot, O.wiper))
    pick_tool = GroundSkill(skill=S.PICK_WIPER, objects=(O.robot, O.wiper))
    model = Tossing3DPracticeModel(
        domain=SWEEP_TABLE_DOMAIN,
        ground_skills=(reset, sweep, return_tool, pick_tool),
        linear_cost_lambda=0.1,
    )
    before = _atoms(on_table=(), in_box=O.cubes[:4], on_floor=O.cubes[4:], active=False)
    outcomes = model.outcomes(
        environment_state=make_tossing3d_search_state(state=belief, true_atoms=before),
        state=belief,
        action=reset,
    )
    probability = mean_competence(belief=belief.skill_beliefs["reset_cubes"])
    assert [weight for weight, _, _ in outcomes] == pytest.approx([probability, 1.0 - probability])
    _, succeeded, after = outcomes[0]
    assert GroundAtom(predicate=TASK_ACTIVE, objects=()) not in after
    assert GroundAtom(predicate=HOLDING_WIPER, objects=(O.robot, O.wiper)) in after
    assert not any(atom.predicate == IN_BOX for atom in after)
    assert model.get_valid_actions(
        environment_state=make_tossing3d_search_state(state=succeeded, true_atoms=after)
    ) == [return_tool]
    assert succeeded.accumulated_cost == pytest.approx(3.0)
    assert outcomes[1][1].accumulated_cost == pytest.approx(3.0)
    assert outcomes[1][2] == before
    assert model.J(belief_state=succeeded, summed_cost=3.0, num_samples=1) == pytest.approx(
        model.J(belief_state=succeeded, summed_cost=0.0, num_samples=1) - 0.3
    )
    for action, next_action in ((return_tool, pick_tool), (pick_tool, sweep)):
        _, succeeded, after = model.outcomes(
            environment_state=make_tossing3d_search_state(state=succeeded, true_atoms=after),
            state=succeeded,
            action=action,
        )[0]
        assert GroundAtom(predicate=TASK_ACTIVE, objects=()) in after
        assert model.get_valid_actions(
            environment_state=make_tossing3d_search_state(state=succeeded, true_atoms=after)
        ) == ([sweep, return_tool] if next_action == sweep else [next_action])


@pytest.mark.parametrize("solver", ["expectimax", "determinized_astar"])
@pytest.mark.parametrize("human_reset", [False, True])
def test_method_builds_from_table_provider_with_optional_human_skills(*, solver, human_reset):
    env = SymbolicEnvironment()
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=SweepTable3DSkillProvider(env=env, offer_human_reset=human_reset),
        pomdp_domain=SWEEP_TABLE_DOMAIN,
        pomdp_solver=solver,
        pomdp_search_depth=2,
        pomdp_num_particles=8,
        seed=0,
    )
    assert set(TABLE_ROBOT_SKILLS) <= set(method.pomdp_state.skill_beliefs)
    human = {"reset_cubes", "reset_wiper"}
    assert (human <= set(method.pomdp_state.skill_beliefs)) is human_reset
    assert (human <= method.pomdp_domain.failure_effect_skills) is human_reset
    assert (
        skill_belief_model(ground_skill=_sweep(), domain=method.pomdp_domain).belief_name
        == "SweepTable"
    )


@pytest.mark.parametrize(
    ("reset_cost", "reset_competence", "choose_reset"),
    [(0.0, 1.0, True), (3.0, 1.0, False), (0.0, 0.0, False), (1.0, 0.25, False), (1.0, 1.0, True)],
)
def test_practice_planner_trades_cube_reset_cost_and_reliability_against_joint_sweep_learning(
    *, reset_cost, reset_competence, choose_reset
):
    beliefs = {
        name: WeightedHypothesisBelief(
            hypotheses=(
                WeightedHypothesis(
                    hypothesis=SkillHypothesis(
                        competence=(
                            0.1
                            if name == "SweepTable"
                            else reset_competence
                            if name == "reset_cubes"
                            else 1.0
                            if name == "PickWiper"
                            else 0.0
                        ),
                        learning_rate=0.2 if name == "SweepTable" else 0.0,
                    ),
                    probability=1.0,
                ),
            )
        )
        for name, _ in SWEEP_TABLE_DOMAIN.belief_sources
    }
    belief = Tossing3DBeliefState(skill_beliefs=beliefs)
    reset = GroundSkill(
        skill=S.RESET_CUBES.model_copy(update={"practice_cost": reset_cost}),
        objects=(O.robot, O.box, O.wiper, *O.cubes),
    )
    sweep = _sweep().model_copy(
        update={"skill": S.SWEEP_TABLE.model_copy(update={"practice_cost": 0.0})}
    )
    pick_tool = GroundSkill(
        skill=S.PICK_WIPER.model_copy(update={"practice_cost": 0.0}),
        objects=(O.robot, O.wiper),
    )
    model = Tossing3DPracticeModel(
        domain=SWEEP_TABLE_DOMAIN,
        ground_skills=(reset, pick_tool, sweep),
        exploration_epsilon=0.0,
        linear_cost_lambda=0.1,
    )
    before = _atoms(
        on_table=(), in_box=O.cubes[:4], on_floor=O.cubes[4:], active=False, holding_wiper=False
    )
    value, action = ExpectimaxPlanner(use_model_j=True).solve(
        environment_state=make_tossing3d_search_state(state=belief, true_atoms=before),
        belief_state=belief,
        summed_cost=0.0,
        horizon=3,
        model=model,
        num_samples=1,
    )
    assert action == (reset if choose_reset else STOP_ACTION)
    expected = 0.1 + reset_competence * 0.2 - 0.1 * reset_cost if choose_reset else 0.1
    assert value == pytest.approx(expected)
