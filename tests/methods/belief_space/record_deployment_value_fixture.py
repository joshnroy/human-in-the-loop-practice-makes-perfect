"""Record `data/recorded_deployment_values.json` for `test_deployment_value_iteration.py`.

Every entry goes through a practice domain's public surface only, so the same code
measures the implementation before and after the deployment values moved onto one
shared value iteration. It was run on main @ 379a51467, where Tossing3D's value was a
hand-written three-state recursion and the shelf and rack each had their own loop:

    scripts/with_env.sh python -m tests.methods.belief_space.record_deployment_value_fixture

Scalars are stored as `float.hex()` and arrays as the sha256 of their float64 bytes, so
equality with the fixture is equality of every bit.
"""

import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from hitl_pmp.methods.belief_space.competence_inference import (
    InferenceConfig,
    create_bayesian_prior,
)
from hitl_pmp.methods.belief_space.practice_domain import (
    SHELF_DEPLOYMENT_SKILLS,
    SWEEP_OFF_SHELF_DOMAIN,
    TOSSING3D_DOMAIN,
    PracticeDomain,
)
from hitl_pmp.methods.belief_space.rack_serving_domain import (
    RACK_ROBOT_SKILLS,
    RACK_SERVING_DOMAIN,
)
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    TOSS_SKILL,
)
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    make_default_tossing3d_belief,
)
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.skill_belief import SkillBelief, SkillHypothesis
from hitl_pmp.methods.belief_space.types.theta import Tossing3DTheta

FIXTURE = Path(__file__).parent / "data" / "recorded_deployment_values.json"
# 4 is every Tossing/kitchen run's horizon, 3 the practice-makes-perfect CLI's.
TOSSING_HORIZONS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 25)
TOSSING_SKILLS = (PICK_SKILL, TOSS_SKILL, OPEN_GRIPPER_SKILL)
# (successes, failures) observed per skill before the belief is valued.
OUTCOME_PATTERNS = (
    ((0, 0), (0, 0), (0, 0)),
    ((1, 1), (1, 1), (1, 1)),
    ((5, 0), (0, 5), (2, 2)),
    ((0, 4), (6, 1), (0, 3)),
    ((9, 2), (3, 7), (4, 0)),
)
GRID = np.linspace(0.0, 1.0, 11)
# 12 and 14 are the shelf's and the rack's own deployment horizons.
TABLE_HORIZONS = (0, 1, 2, 3, 6, 12, 14, 20)


class RecordedDeploymentValues:
    """The recorded quantities, computed the same way by recorder and test."""

    @staticmethod
    def digest(*, values: Any) -> str:
        return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()

    @staticmethod
    def conditioned(*, belief: SkillBelief, successes: int, failures: int) -> SkillBelief:
        for _ in range(successes):
            belief = belief.condition_outcome(success=True)
        for _ in range(failures):
            belief = belief.condition_outcome(success=False)
        return belief

    @staticmethod
    def tossing_states() -> dict[str, Tossing3DBeliefState]:
        states = {}
        for model, engine, (index, pattern) in product(
            ("global_curve", "local_trend"), ("particle", "grid"), enumerate(OUTCOME_PATTERNS)
        ):
            beliefs = {
                name: RecordedDeploymentValues.conditioned(
                    belief=create_bayesian_prior(
                        model=model,  # type: ignore[arg-type]
                        engine=engine,  # type: ignore[arg-type]
                        seed=index * 3 + slot,
                        num_particles=64,
                    ),
                    successes=successes,
                    failures=failures,
                )
                for slot, (name, (successes, failures)) in enumerate(
                    zip(TOSSING_SKILLS, pattern, strict=True)
                )
            }
            states[f"{model}-{engine}-{index}"] = Tossing3DBeliefState(
                skill_beliefs=beliefs, pending_examples={TOSS_SKILL: index}
            )
        return states

    @staticmethod
    def tossing() -> dict[str, object]:
        recorded: dict[str, object] = {}
        states = RecordedDeploymentValues.tossing_states()
        for horizon in TOSSING_HORIZONS:
            domain = TOSSING3D_DOMAIN.model_copy(update={"deployment_horizon": horizon})
            recorded[f"theta-grid/h{horizon}"] = RecordedDeploymentValues.digest(
                values=[
                    domain.evaluate_theta(
                        theta=Tossing3DTheta(
                            pick=SkillHypothesis(competence=pick, learning_rate=0.0),
                            toss=SkillHypothesis(competence=toss, learning_rate=0.0),
                            open_gripper=SkillHypothesis(competence=opened, learning_rate=0.0),
                        )
                    )
                    for pick, toss, opened in product(GRID, GRID, GRID)
                ]
            )
            for name, state in states.items():
                recorded[f"samples/{name}/h{horizon}"] = RecordedDeploymentValues.digest(
                    values=domain.sample_policy_values(
                        beliefs=state.skill_beliefs, rng=np.random.default_rng(7), count=4096
                    )
                )
                for cost_lambda in (None, 0.0003):
                    model = Tossing3DPracticeModel(
                        deployment_horizon=horizon, linear_cost_lambda=cost_lambda
                    )
                    recorded[f"J/{name}/h{horizon}/lambda{cost_lambda}"] = model.J(
                        belief_state=state, summed_cost=7.0, num_samples=1
                    ).hex()
        return recorded

    @staticmethod
    def table_states(*, domain: PracticeDomain) -> dict[str, Tossing3DBeliefState]:
        states = {
            "production-prior": make_default_tossing3d_belief(
                domain=domain, model="local_trend", engine="grid"
            ),
            "coarse-prior": make_default_tossing3d_belief(
                domain=domain,
                model="local_trend",
                engine="grid",
                num_particles=32,
                inference_config=InferenceConfig(competence_bins=9, learning_rate_bins=4),
            ),
            "particle-prior": make_default_tossing3d_belief(
                domain=domain, model="global_curve", engine="particle", seed=3
            ),
        }
        coarse = states["coarse-prior"]
        states["coarse-trained"] = coarse.model_copy(
            update={
                "skill_beliefs": {
                    name: RecordedDeploymentValues.conditioned(
                        belief=belief, successes=index % 4, failures=(index * 2) % 3
                    )
                    for index, (name, belief) in enumerate(coarse.skill_beliefs.items())
                }
            }
        )
        return states

    @staticmethod
    def table_domain(
        *, domain: PracticeDomain, skills: tuple[str, ...], label: str
    ) -> dict[str, object]:
        recorded: dict[str, object] = {}
        for name, state in RecordedDeploymentValues.table_states(domain=domain).items():
            recorded[f"{label}/J/{name}"] = (
                Tossing3DPracticeModel(domain=domain, linear_cost_lambda=0.0003)
                .J(belief_state=state, summed_cost=7.0, num_samples=1)
                .hex()
            )
            for horizon in TABLE_HORIZONS:
                at_horizon = domain.model_copy(update={"deployment_horizon": horizon})
                recorded[f"{label}/expected/{name}/h{horizon}"] = at_horizon.expected_policy_value(
                    beliefs=state.skill_beliefs
                ).hex()
                recorded[f"{label}/samples/{name}/h{horizon}"] = RecordedDeploymentValues.digest(
                    values=at_horizon.sample_policy_values(
                        beliefs=state.skill_beliefs, rng=np.random.default_rng(11), count=1024
                    )
                )
        rng = np.random.default_rng(5)
        draws = rng.uniform(size=(256, len(skills)))
        draws[:8] = np.eye(8, len(skills))
        draws[8:16] = 1.0 - np.eye(8, len(skills))
        for horizon in TABLE_HORIZONS:
            at_horizon = domain.model_copy(update={"deployment_horizon": horizon})
            recorded[f"{label}/theta/h{horizon}"] = RecordedDeploymentValues.digest(
                values=[
                    at_horizon.evaluate_theta(
                        theta={
                            name: SkillHypothesis(competence=float(value), learning_rate=0.0)
                            for name, value in zip(skills, row, strict=True)
                        }
                    )
                    for row in draws
                ]
            )
        return recorded

    @staticmethod
    def compute() -> dict[str, object]:
        return {
            **RecordedDeploymentValues.tossing(),
            **RecordedDeploymentValues.table_domain(
                domain=SWEEP_OFF_SHELF_DOMAIN, skills=SHELF_DEPLOYMENT_SKILLS, label="shelf"
            ),
            **RecordedDeploymentValues.table_domain(
                domain=RACK_SERVING_DOMAIN, skills=RACK_ROBOT_SKILLS, label="rack"
            ),
        }


if __name__ == "__main__":
    FIXTURE.write_text(json.dumps(RecordedDeploymentValues.compute(), indent=1) + "\n")
