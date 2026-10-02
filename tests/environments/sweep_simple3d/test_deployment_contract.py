"""The generic shared-competence model supports five separately grounded sweeps."""

from hitl_pmp.environments.sweep_simple3d.symbolic import SIMPLE_PREDICATES, SimpleSymbols
from hitl_pmp.methods.belief_space.sweep_deployment_model import SweepDeploymentExpectation
from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis, WeightedHypothesis
from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import WeightedHypothesisBelief
from hitl_pmp.planning.grounding import SkillGrounder


def test_exact_model_requires_pick_and_all_five_cube_sweeps() -> None:
    atoms = SkillGrounder.all_possible_ground_atoms(
        objects=SimpleSymbols.objects(), predicates=tuple(SIMPLE_PREDICATES.values())
    )
    skills = tuple(
        sorted(
            SkillGrounder.applicable_ground_skills(
                skills=tuple(
                    s for s in SimpleSymbols.skills() if s.name in SimpleSymbols.TRAINABLE
                ),
                objects=SimpleSymbols.objects(),
                true_atoms=atoms,
            ),
            key=str,
        )
    )
    beliefs = {
        name: WeightedHypothesisBelief(
            hypotheses=(
                WeightedHypothesis(
                    hypothesis=SkillHypothesis(competence=1.0, learning_rate=0.0), probability=1.0
                ),
            )
        )
        for name in SimpleSymbols.TRAINABLE
    }
    for horizon, expected in ((5, 0.0), (6, 1.0), (10, 1.0)):
        model = SweepDeploymentExpectation(
            ordered_skills=skills,
            initial_atoms=SimpleSymbols.initial_atoms(),
            goal_atoms=SimpleSymbols.goal_atoms(),
            horizon=horizon,
        )
        assert model.evaluate(beliefs=beliefs) == expected
