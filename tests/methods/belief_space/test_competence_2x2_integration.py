"""Exercise the four competence models through the real method and CLI boundaries."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DOracle, Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.types import Tossing3DState
from hitl_pmp.methods.belief_space.competence_inference import BayesianSkillBelief
from hitl_pmp.methods.belief_space.tossing3d_constants import PICK_SKILL, RESET_SKILL, TOSS_SKILL
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_observation_model import refit_belief_state
from hitl_pmp.planning.grounding import SkillGrounder

Model = Literal["global_curve", "local_trend"]
Engine = Literal["particle", "grid"]
ARMS: tuple[tuple[Model, Engine], ...] = (
    ("global_curve", "particle"),
    ("global_curve", "grid"),
    ("local_trend", "particle"),
    ("local_trend", "grid"),
)


def _build(
    *,
    model: Model,
    engine: Engine,
    decision_log: Path | None = None,
    cost_lambda: float | None = None,
) -> Tossing3DPomdpMethod:
    env = Tossing3DEnvironment(scene_bg=False)
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env, human_reset_practice_cost=0.25),
        seed=17,
        pomdp_competence_model=model,
        pomdp_inference_engine=engine,
        pomdp_num_particles=64,
        pomdp_grid_competence_bins=5,
        pomdp_grid_learning_rate_bins=4,
        pomdp_search_depth=1,
        pomdp_num_samples=2,
        pomdp_linear_cost_lambda=cost_lambda,
        sampler_max_train_iters=1,
        decision_log=decision_log,
    )
    # No action is in flight in these lifecycle tests. A valid current state lets
    # end_cycle flush through its normal path without constructing a simulator.
    env.current_state = Tossing3DState(
        data={obj: np.zeros(obj.type.dim) for obj in method.objects()},
        abstract_atoms=frozenset(),
    )
    return method


def _ground(*, method: Tossing3DPomdpMethod, name: str) -> GroundSkill:
    universe = SkillGrounder.all_possible_ground_atoms(
        objects=method.objects(), predicates=method.predicates()
    )
    return next(
        ground
        for ground in SkillGrounder.applicable_ground_skills(
            skills=(*method.skills(), *method.human_skills()),
            objects=method.objects(),
            true_atoms=universe,
        )
        if ground.skill.name == name
    )


def _belief(*, method: Tossing3DPomdpMethod, name: str) -> BayesianSkillBelief:
    belief = method.pomdp_state.skill_beliefs[name]
    assert isinstance(belief, BayesianSkillBelief)
    return belief


@pytest.mark.parametrize(("model", "engine"), ARMS)
def test_cli_constructs_and_runs_each_selected_inference_arm(
    *, model: Model, engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    constructed: list[Tossing3DPomdpMethod] = []

    def run_method(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[DomainContext], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None:
        assert num_cycles == 10
        assert max_steps_per_interaction > 0
        assert args.defer_rendering is True
        env = Tossing3DEnvironment(scene_bg=False)
        method = method_factory(
            DomainContext(
                env=env,
                skill_provider=Tossing3DSkillProvider(env=env),
                oracle=Tossing3DOracle(env=env),
            )
        )
        assert isinstance(method, Tossing3DPomdpMethod)
        constructed.append(method)
        for belief in method.pomdp_state.skill_beliefs.values():
            assert isinstance(belief, BayesianSkillBelief)
            assert (belief.model_name, belief.engine) == (model, engine)
            assert belief.config.competence_bins == 5
            assert belief.config.learning_rate_bins == 4
        pick = _ground(method=method, name=PICK_SKILL)
        before = _belief(method=method, name=PICK_SKILL).mean_competence()
        method.observe_outcome(ground_skill=pick, success=True)
        assert _belief(method=method, name=PICK_SKILL).mean_competence() > before
        selected = method.select_skill_to_practice(true_atoms=pick.preconditions)
        assert len(selected) == 1
        assert method.env._backend is None  # noqa: SLF001

    monkeypatch.setattr(Tossing3DCli, "run_method", run_method)
    Cli.main(
        argv=[
            "--env",
            "tossing3d",
            "--method",
            "pomdp",
            "--pomdp-competence-model",
            model,
            "--pomdp-inference-engine",
            engine,
            "--pomdp-num-particles",
            "64",
            "--pomdp-grid-competence-bins",
            "5",
            "--pomdp-grid-learning-rate-bins",
            "4",
            "--pomdp-search-depth",
            "1",
            "--pomdp-num-samples",
            "2",
            "--num-cycles",
            "10",
        ]
    )
    assert len(constructed) == 1


@pytest.mark.parametrize("cost_lambda", [None, 0.07])
def test_cost_observations_dispatch_charges_and_objective_are_identical_across_arms(
    *, cost_lambda: float | None, tmp_path: Path
) -> None:
    costs_by_arm = []
    logs_by_arm = []
    objectives_by_arm = []
    for model, engine in ARMS:
        decision_log = tmp_path / f"{model}-{engine}.jsonl"
        method = _build(
            model=model, engine=engine, cost_lambda=cost_lambda, decision_log=decision_log
        )
        initial_costs = {
            name: _belief(method=method, name=name).cost_belief.signature()
            for name in method.pomdp_state.skill_beliefs
        }
        for name, success, random in (
            (PICK_SKILL, True, False),
            (TOSS_SKILL, False, True),
            (TOSS_SKILL, True, False),
        ):
            ground = _ground(method=method, name=name)
            before = method.pomdp_state
            method.record_action_cost(ground_skill=ground)
            assert method.pomdp_state.skill_beliefs == before.skill_beliefs
            method.observe_outcome(
                ground_skill=ground, success=success, was_random_exploration=random
            )
        reset = _ground(method=method, name=RESET_SKILL)
        method.record_action_cost(ground_skill=reset)
        method.observe_help_granted(state=method.env.get_current_state())
        assert method.pomdp_state.accumulated_cost == 3.25
        conditioned_costs = {
            name: _belief(method=method, name=name).cost_belief.signature()
            for name in method.pomdp_state.skill_beliefs
        }
        method.end_cycle()
        assert method.pomdp_state.accumulated_cost == 3.25
        assert conditioned_costs == {
            name: _belief(method=method, name=name).cost_belief.signature()
            for name in method.pomdp_state.skill_beliefs
        }
        assert conditioned_costs[PICK_SKILL] != initial_costs[PICK_SKILL]
        costs_by_arm.append((initial_costs, conditioned_costs))
        logs_by_arm.append([
            (event["skill"], event["configured_cost_observation"], event["cost_observation_source"])
            for event in map(json.loads, decision_log.read_text().splitlines())
            if event["event"] == "outcome"
        ])
        objectives_by_arm.append([
            method._pomdp_model.G(policy_value=0.6, summed_cost=cost)  # noqa: SLF001
            for cost in (0.0, 3.25, 21.0)
        ])
    assert all(costs == costs_by_arm[0] for costs in costs_by_arm)
    assert all(logs == logs_by_arm[0] for logs in logs_by_arm)
    assert [cost for _, cost, _ in logs_by_arm[0]] == [1.0, 1.0, 1.0, 0.25]
    assert all(values == objectives_by_arm[0] for values in objectives_by_arm)
    expected = (
        [0.6, 0.6, -np.inf]
        if cost_lambda is None
        else [0.6 - cost_lambda * c for c in (0, 3.25, 21)]
    )
    assert objectives_by_arm[0] == pytest.approx(expected)


@pytest.mark.parametrize(("model", "engine"), ARMS)
def test_random_toss_is_sampler_data_and_cost_evidence_but_never_competence_evidence(
    *, model: Model, engine: Engine
) -> None:
    method = _build(model=model, engine=engine)
    toss = _ground(method=method, name=TOSS_SKILL)
    before = _belief(method=method, name=TOSS_SKILL)
    method.observe_outcome(ground_skill=toss, success=False, was_random_exploration=True)
    after_random = _belief(method=method, name=TOSS_SKILL)
    assert after_random.latent_values == before.latent_values
    assert after_random.state_weights == before.state_weights
    assert after_random.cycle_successes == after_random.cycle_failures == 0
    assert after_random.cost_belief != before.cost_belief
    assert method.pomdp_state.pending_examples.get(TOSS_SKILL, 0) == 0
    method.observe_sampler_outcome(
        skill_name=TOSS_SKILL, param_dim=4, sampler_input=[0.0], success=False
    )
    sampler = method.sampler(skill_name=TOSS_SKILL, param_dim=4)
    assert sampler.num_observations == 1
    assert method.pomdp_state.pending_examples[TOSS_SKILL] == 1
    assert _belief(method=method, name=TOSS_SKILL) == after_random
    method.end_cycle()
    advanced = _belief(method=method, name=TOSS_SKILL)
    assert advanced.total_training_examples == advanced.incoming_training_examples == 0
    # Every real boundary now consumes one process-noise stream, n = 0 included.
    assert advanced.process_transition_count == 1
    assert method.pomdp_state.sampler_training[TOSS_SKILL].failures == 1
    assert advanced.cycle_successes == advanced.cycle_failures == 0
    assert sampler.is_fitted


@pytest.mark.parametrize(("model", "engine"), ARMS)
def test_hypothetical_and_real_one_class_refits_preserve_competence_without_mutation(
    *, model: Model, engine: Engine
) -> None:
    method = _build(model=model, engine=engine)
    method.observe_outcome(ground_skill=_ground(method=method, name=PICK_SKILL), success=True)
    method.observe_outcome(ground_skill=_ground(method=method, name=TOSS_SKILL), success=False)
    method.observe_sampler_outcome(
        skill_name=TOSS_SKILL, param_dim=4, sampler_input=[1.0], success=False
    )
    before = method.pomdp_state.model_dump(mode="json")
    forecast = refit_belief_state(state=method.pomdp_state)
    assert method.pomdp_state.model_dump(mode="json") == before
    assert method._belief_history == {}  # noqa: SLF001
    assert method.sampler(skill_name=TOSS_SKILL, param_dim=4).is_fitted is False
    method.end_cycle()
    for name in (PICK_SKILL, TOSS_SKILL):
        actual = _belief(method=method, name=name)
        predicted = forecast.skill_beliefs[name]
        assert isinstance(predicted, BayesianSkillBelief)
        # The carve-out made visible: the search forecast kept the zero-example
        # identity while the real boundary applied the n = 0 noise step, so the
        # two now differ in latents while agreeing on every learning total.
        assert (actual.latent_values, actual.state_weights) != (
            predicted.latent_values,
            predicted.state_weights,
        )
        assert actual.total_training_examples == 0
        assert actual.process_transition_count == 1
        assert predicted.process_transition_count == 0
    assert method.pomdp_state.pending_examples == {}
    assert method.sampler(skill_name=TOSS_SKILL, param_dim=4).is_fitted


@pytest.mark.parametrize(("model", "engine"), ARMS)
def test_cycle_logs_preserve_sf_history_and_zero_example_cycles_for_smoothing(
    *, model: Model, engine: Engine, tmp_path: Path
) -> None:
    decision_log = tmp_path / "decisions.jsonl"
    method = _build(model=model, engine=engine, decision_log=decision_log)
    pick = _ground(method=method, name=PICK_SKILL)
    method.observe_outcome(ground_skill=pick, success=True)
    method.end_cycle()
    after_learning = _belief(method=method, name=PICK_SKILL)
    method.end_cycle()
    after_empty = _belief(method=method, name=PICK_SKILL)
    # The idle boundary applies the n = 0 noise step, so the latents move while
    # costs and learning totals stand still.
    assert (after_empty.latent_values, after_empty.state_weights) != (
        after_learning.latent_values,
        after_learning.state_weights,
    )
    assert after_empty.cost_belief == after_learning.cost_belief
    assert after_empty.total_training_examples == 0
    assert after_empty.process_transition_count == 2
    assert after_empty.incoming_training_examples == 0
    assert after_empty.cycle_index == 2
    method.observe_outcome(ground_skill=pick, success=False)
    method.end_cycle()
    history = method._belief_history[PICK_SKILL]  # noqa: SLF001
    assert [(b.cycle_successes, b.cycle_failures) for b in history] == [(1, 0), (0, 0), (0, 1)]
    assert [b.cycle_index for b in history] == [0, 1, 2]
    assert [b.total_training_examples for b in history] == [0, 0, 0]
    assert [b.incoming_training_examples for b in history] == [0, 0, 0]
    events = list(map(json.loads, decision_log.read_text().splitlines()))
    refits = [event for event in events if event["event"] == "refit"]
    assert [event["training_examples"] for event in refits] == [{}, {}, {}]
    assert all(event["learning_rate_evidence"] == "success_failure_only" for event in refits)
    assert all("learning_rate_observations" not in event for event in events)
    smoothing = [event for event in events if event["event"] == "smoothing"]
    assert len(smoothing) == 3
    for index, event in enumerate(smoothing):
        assert event["use"] == "retrospective_diagnostics_only"
        assert (event["competence_model"], event["inference_engine"]) == (model, engine)
        cycles = event["history"][PICK_SKILL]
        assert [cycle["cycle_index"] for cycle in cycles] == list(range(index + 1))
        assert cycles[-1]["filtered_competence"] == history[index].mean_competence()
        assert cycles[-1]["smoothed_competence"] == pytest.approx(history[index].mean_competence())
    assert method.env._backend is None  # noqa: SLF001
