"""Launch one step-normalized expectimax + coding skill learner."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from hitl_pmp.agentic_runtime.sandbox import SandboxSettings
from hitl_pmp.cli import Cli
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.practice_costs import PracticeCosts
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli as DomainCli
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DOracle, Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.state_log import StateLogHeader, StateLogWriter
from hitl_pmp.full_agentic.runner import FullAgenticRunner
from hitl_pmp.methods.practice_makes_perfect.cli import Tossing3DPomdpCli
from hitl_pmp.planning.fast_downward import FastDownwardPlanner
from hitl_pmp.step_protocol import StepFiles, StepPracticeRunner

from .artifacts import ModelBudgetExhausted, SkillBundle
from .execution import HybridEnvironment, HybridEvaluation, HybridSnapshot, SkillExecutor
from .learner import SkillLearner
from .method import HybridMethod


def main(*, argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hybrid-sandbox", type=Path, required=True)
    parser.add_argument("--model-budget", type=float, default=20)
    parser.add_argument("--hybrid-smoke", action="store_true")
    parser.add_argument("--agentic-states", action="store_true")
    options, rest = parser.parse_known_args(argv)
    args = Cli.parse_args(
        argv=[
            "--env",
            "tossing3d",
            "--method",
            "pomdp",
            "--seed",
            "0",
            "--pomdp-solver",
            "expectimax",
            "--pomdp-search-depth",
            "6",
            "--pomdp-inference-engine",
            "grid",
            "--pomdp-competence-model",
            "local_trend",
            "--pomdp-observation-probability-weight",
            "0",
            "--pomdp-linear-cost-lambda",
            "3e-6",
            "--exploration-epsilon",
            "0",
            "--max-steps-per-interaction",
            "20",
            "--goal-pursuit-horizon",
            "0",
            "--practice-reset-policy",
            "never",
            "--practice-step-budget",
            "85000",
            "--measurement-interval-steps",
            "1700",
            "--evaluation-control-steps",
            "500",
            "--num-test-tasks",
            "10",
            "--stop-after-perfect-evaluations",
            "3",
            "--fragile-object",
            "--mat-size",
            "1",
            "--defer-rendering",
            *rest,
        ]
    )
    args.method = (
        "pomdp-agentic-skills-states" if options.agentic_states else "pomdp-agentic-skills"
    )
    if options.agentic_states and (
        args.pomdp_solver != "expectimax" or args.practice_cost_config is None
    ):
        raise ValueError("LLCC requires expectimax and an explicit normalized cost configuration")
    if options.agentic_states and options.hybrid_smoke:
        raise ValueError(
            "LLCC requires generated states; use the isolated integration fixture for smoke tests"
        )
    # Deployment and the end-of-cycle plan refresh need this, even with expectimax.
    if not options.agentic_states:
        FastDownwardPlanner._fast_downward_script()  # noqa: SLF001 -- fail before paid coding
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Preserve the existing run; choose an empty output directory")
    args.output_dir = output
    settings = SandboxSettings.model_validate(json.loads(options.hybrid_sandbox.read_text()))
    settings = settings.model_copy(update={"artifact_dir": output / "practice_trajectories"})
    sys.path.insert(0, str(settings.robocode_checkout.resolve() / "src"))
    args.hybrid_sandbox = settings.model_dump(mode="json")
    args.hybrid_model_budget = options.model_budget
    args.hybrid_smoke = options.hybrid_smoke
    FullAgenticRunner.validate_runtime(settings=settings)
    costs = PracticeCosts.load(path=args.practice_cost_config)

    class Tossing3DCli:
        @staticmethod
        def add_arguments(*, parser: argparse.ArgumentParser) -> None:
            DomainCli.add_arguments(parser=parser)

        @staticmethod
        def run_method(*, method_factory: Any, **unused: Any) -> None:
            problem = DomainCli.build_practice_problem(args=args)
            env = HybridEnvironment(**problem.env.model_dump())
            env.executor = SkillExecutor(settings=settings)
            problem.env = env
            problem.tasks.env = env
            provider = Tossing3DSkillProvider(
                env=env,
                human_reset_practice_cost=args.human_reset_practice_cost,
                offer_human_reset=args.human_reset,
            )
            context = DomainContext(
                env=env, skill_provider=provider, oracle=Tossing3DOracle(env=env)
            )
            base = method_factory(context)
            config = {key: getattr(base, key) for key in type(base).model_fields}
            method = HybridMethod(**config)
            snapshot_type, evaluation_type = HybridSnapshot, HybridEvaluation
            if options.agentic_states:
                from hitl_pmp.agentic_states.classifier import LanguageClassifier
                from hitl_pmp.agentic_states.execution import StatesEvaluation, StatesSnapshot
                from hitl_pmp.agentic_states.method import SkillsStatesMethod

                method = SkillsStatesMethod(**config)
                classifier_configuration = dict(
                    settings=settings.model_dump(mode="json"),
                    output=str(output / "classifications"),
                    budget_path=str(output / "model_budget.json"),
                    budget_limit=options.model_budget,
                )
                method.classifier_configuration = classifier_configuration
                method.classifier = LanguageClassifier(
                    settings=settings,
                    output=output / "classifications",
                    budget_path=output / "model_budget.json",
                    budget_limit=options.model_budget,
                    phase="practice",
                )
                snapshot_type, evaluation_type = StatesSnapshot, StatesEvaluation
            writer = StateLogWriter(
                output_path=output / "tossing3d_state_log.jsonl",
                header=StateLogHeader(
                    layout=env.layout,
                    variant=args.variant,
                    scene_bg=args.scene_bg,
                    canonical_seed=args.canonical_seed,
                    seed=args.seed,
                    test_env_seed_offset=args.test_env_seed_offset,
                ),
            )
            env.attach_state_log_writer(writer=writer)
            learner = None
            if options.hybrid_smoke:
                # A short motor-only fixture, not a trained initialization or paid model call.
                env.bundle = SkillBundle(
                    files={
                        "skills.py": """
import numpy as np
class GeneratedSkills:
    def __init__(self, *args): pass
    def reset(self, observation, skill): self.k = 0
    def get_action(self, observation):
        self.k += 1
        return np.zeros(18) if self.k <= 3 else None
"""
                    }
                )

                def initialize() -> None:
                    pass

            else:
                learner_type = SkillLearner
                learner_options = {}
                if options.agentic_states:
                    from hitl_pmp.agentic_states.learner import SkillsStatesLearner

                    learner_type = SkillsStatesLearner
                    learner_options = {"on_accept": method.accept_manifest}
                learner = learner_type(
                    **learner_options,
                    output=output,
                    env=env,
                    settings=settings,
                    costs=costs,
                    budget=options.model_budget,
                )
                initialize = learner.revise
                method.revise_skills = learner.revise
            StepFiles.json(
                path=output / "hybrid_protocol.json",
                value=dict(
                    method=args.method,
                    planner="expectimax",
                    depth=args.pomdp_search_depth,
                    representation="generated_language_states"
                    if options.agentic_states
                    else "fixed_structured",
                    learned="robot_skill_code_and_language_states"
                    if options.agentic_states
                    else "robot_skill_code",
                    classifier_model="claude-opus-5-5" if options.agentic_states else None,
                    classifier_effort="low" if options.agentic_states else None,
                    evaluation_classification_budget_scope="separate_uncapped_ledger"
                    if options.agentic_states
                    else "not_applicable",
                    model_budget_scope="coding_and_practice_classification"
                    if options.agentic_states
                    else "coding",
                    deployment_planner="language_graph_shortest_success_path"
                    if options.agentic_states
                    else "fixed_pddl",
                    value_model="existing_pick_toss_open_competence_forecast",
                    prompt="original",
                    bootstrap="fresh_interface_only",
                    seed=args.seed,
                    model="claude-opus-5-5",
                    effort="high",
                    model_budget=options.model_budget,
                    sampler="none",
                    baseline_pick_feasibility=False,
                    exploration_epsilon=0,
                    physical_tests="planner_owned_and_counted",
                    extra_geometry_tools=False,
                    skill_limits=dict(
                        PickCube=400, MoveToTossLocationAndToss=1000, OpenGripper=100
                    ),
                    smoke=options.hybrid_smoke,
                ),
            )
            try:
                StepPracticeRunner.run(
                    args=args,
                    method=method,
                    problem=problem,
                    encode_snapshot=snapshot_type.encode,
                    evaluate_snapshot=evaluation_type.run,
                    initialize=initialize,
                    stop_exceptions=(ModelBudgetExhausted,),
                )
            finally:
                writer.close()
                env.close()

    Tossing3DPomdpCli.run(args=args, env_cli=Tossing3DCli)


if __name__ == "__main__":
    main()
