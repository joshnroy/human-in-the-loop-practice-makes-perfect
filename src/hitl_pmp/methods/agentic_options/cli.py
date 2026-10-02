"""Composition of external experiment inputs, isolated runtime and search."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from hitl_pmp.agentic_runtime.learner import CodePolicyLearner
from hitl_pmp.agentic_runtime.observations import ObservationEvidence, ObservationMode
from hitl_pmp.agentic_runtime.robocode import CodingSettings, RobocodeCodingAgent
from hitl_pmp.agentic_runtime.sandbox import DockerContainer, DockerPolicyExecutor
from hitl_pmp.agentic_runtime.types import GeneratedLibrary, OptionContract
from hitl_pmp.agentic_runtime.vision import RobocodeVisionClient, VLMJudge
from hitl_pmp.cli_protocols import EnvironmentCli
from hitl_pmp.environments.tossing3d.agentic_bridge import (
    AgenticTossing3DEnvironment,
    Tossing3DAgenticBridge,
)
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.kinder_backend import ControllerRun
from hitl_pmp.environments.tossing3d.problem import Tossing3DProblem
from hitl_pmp.environments.tossing3d.renderer import Tossing3DRenderer
from hitl_pmp.method_runner import MethodRunner
from hitl_pmp.methods.belief_space.tossing3d_observation_model import make_default_tossing3d_belief
from hitl_pmp.practice_loop import PracticeResetPolicy

from .belief import TossingLearnerBelief
from .chain_model import SkillChainModel
from .method import AgenticOptionsMethod
from .types import SearchConfig


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    coding: CodingSettings
    vision: dict[str, Any]

    def resolved_vision(self, *, artifact_dir: Path) -> dict[str, Any]:
        config = dict(self.vision)
        if config.get("provider") == "robocode_broker":
            config.setdefault("backend", self.coding.backend)
            config.setdefault("model", self.coding.model)
            config.setdefault("reasoning_effort", self.coding.reasoning_effort)
            config["robocode_checkout"] = str(self.coding.sandbox.robocode_checkout.resolve())
            config["artifact_dir"] = str(artifact_dir.resolve())
        return config


class AgenticOptionsCli:
    """No domain prompt is embedded in the method implementation."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--agentic-inputs", type=Path, required=True)
        parser.add_argument("--agentic-runtime-config", type=Path, required=True)
        parser.add_argument("--agentic-library", type=Path)
        parser.add_argument("--num-cycles", type=int, default=2)
        parser.add_argument("--max-steps-per-interaction", type=int, default=10)
        parser.add_argument("--agentic-search-depth", type=int, default=3)
        parser.add_argument("--agentic-num-particles", type=int, default=256)
        parser.add_argument("--agentic-observation-probability-weight", type=float, default=0.1)
        parser.add_argument("--agentic-linear-cost-lambda", type=float)

    @staticmethod
    def run(*, args: argparse.Namespace, env_cli: type[EnvironmentCli]) -> None:
        del env_cli
        if args.env != "tossing3d":
            raise ValueError("The initial agentic option composition supports Tossing3D only")
        if args.output_dir is None:
            raise ValueError("Agentic runs require --output-dir for source and judgment provenance")
        if args.practice_reset_policy != PracticeResetPolicy.NEVER:
            raise ValueError("Agentic Tossing3D requires --practice-reset-policy never")
        if args.practice_reset_interval is not None:
            raise ValueError("Agentic practice does not allow scheduled resets")
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        runtime = RuntimeConfig.model_validate_json(args.agentic_runtime_config.read_text())
        runtime = runtime.model_copy(
            update={
                "coding": runtime.coding.model_copy(
                    update={
                        "sandbox": runtime.coding.sandbox.model_copy(
                            update={
                                "seed": args.seed,
                                "artifact_dir": output / "policy-runtime",
                                "image": DockerContainer(
                                    settings=runtime.coding.sandbox
                                ).image_id(),
                            }
                        ),
                        "artifact_dir": output / "coding-agent",
                    }
                ),
            }
        )
        runtime.vision = runtime.resolved_vision(artifact_dir=output / "vision-runtime")
        (output / "runtime_provenance.json").write_text(
            json.dumps(
                {
                    "coding": runtime.coding.model_dump(mode="json"),
                    "vision": {
                        key: runtime.vision.get(key) for key in ("provider", "backend", "model")
                    },
                },
                indent=2,
            )
        )
        config = SearchConfig(
            depth=args.agentic_search_depth,
            linear_cost_lambda=args.agentic_linear_cost_lambda,
            num_particles=args.agentic_num_particles,
            observation_probability_weight=getattr(
                args, "agentic_observation_probability_weight", 0.1
            ),
            seed=args.seed,
        )
        bundle = json.loads((args.agentic_inputs / "bundle.json").read_text())
        prompts = {
            name: (args.agentic_inputs / filename).read_text()
            for name, filename in bundle["prompts"].items()
        }
        domain_inputs = "\n\n".join(
            (args.agentic_inputs / bundle[name]).read_text() for name in ("task", "robot_api")
        )
        domain_inputs += "\n\nResolved experiment settings:\n" + json.dumps(
            {
                **bundle,
                "configured_costs": {
                    "robot": bundle["configured_costs"]["robot"],
                    "human": args.human_reset_practice_cost,
                },
                "human_enabled": args.human_reset,
                "seed": args.seed,
            },
            indent=2,
        )
        (output / "input_bundle.json").write_text(json.dumps(bundle, indent=2))
        (output / "input_prompts.json").write_text(json.dumps(prompts, indent=2))
        (output / "domain_inputs.md").write_text(domain_inputs)
        practice = AgenticOptionsCli.adapt_problem(
            problem=Tossing3DCli.build_practice_problem(args=args)
        )
        evaluation = AgenticOptionsCli.adapt_problem(
            problem=Tossing3DCli.build_evaluation_problem(args=args)
        )
        try:
            practice.hard_reset()
            AgenticOptionsCli.run_initialized(
                args=args,
                runtime=runtime,
                config=config,
                bundle=bundle,
                prompts=prompts,
                domain_inputs=domain_inputs,
                output=output,
                practice=practice,
                evaluation=evaluation,
            )
        finally:
            practice.env.close()
            evaluation.env.close()

    @staticmethod
    def run_initialized(
        *,
        args: argparse.Namespace,
        runtime: RuntimeConfig,
        config: SearchConfig,
        bundle: dict[str, Any],
        prompts: dict[str, str],
        domain_inputs: str,
        output: Path,
        practice: Tossing3DProblem,
        evaluation: Tossing3DProblem,
    ) -> None:
        """Generate against the spawned world, then keep it throughout practice."""
        observation_mode: ObservationMode = bundle["observation_mode"]
        assert isinstance(practice.env, AgenticTossing3DEnvironment)
        assert isinstance(evaluation.env, AgenticTossing3DEnvironment)
        practice_observer = Tossing3DAgenticBridge(
            env=practice.env, observation_mode=observation_mode
        )
        evaluation_observer = Tossing3DAgenticBridge(
            env=evaluation.env, observation_mode=observation_mode
        )
        human_input = json.loads((args.agentic_inputs / bundle["human_input"]).read_text())
        for key in ("HUMAN_TASK_QUESTION", "HUMAN_TASK_RESPONSE"):
            if not isinstance(human_input.get(key), str) or not human_input[key].strip():
                raise ValueError(f"Human input requires a nonempty {key} constant")
        input_files = {
            "human_input.json": json.dumps(human_input, indent=2, allow_nan=False),
            "initial_observation.json": json.dumps(
                practice_observer.observe(), indent=2, allow_nan=False
            ),
            "robot_spec.json": json.dumps(
                practice_observer.robot_spec(), indent=2, allow_nan=False
            ),
        }
        for name, content in input_files.items():
            (output / name).write_text(content)
        print(human_input["HUMAN_TASK_QUESTION"], flush=True)
        print("Fixed experiment response: " + human_input["HUMAN_TASK_RESPONSE"], flush=True)
        agent = RobocodeCodingAgent(settings=runtime.coding, input_files=input_files)
        library = (
            GeneratedLibrary.model_validate_json(args.agentic_library.read_text())
            if args.agentic_library is not None
            else agent.generate(prompt=prompts["generate"] + "\n\n" + domain_inputs)
        )
        library = AgenticOptionsCli.configure_library(
            library=library,
            bundle=bundle,
            human_enabled=args.human_reset,
            human_cost=args.human_reset_practice_cost,
        )
        (output / "initial_library.json").write_text(library.model_dump_json(indent=2))
        print(
            "Generated skills and abstract states:\n" + library.manifest.model_dump_json(indent=2),
            flush=True,
        )
        learner = CodePolicyLearner(
            library=library,
            agent=agent,
            improvement_prompt=prompts["improve"] + "\n\n" + domain_inputs,
            artifact_dir=output / "learner",
        )
        judge = VLMJudge(
            client=RobocodeVisionClient(config=runtime.vision),
            classification_prompt=prompts["classify"],
            option_check_prompt=prompts["verify_option"],
        )
        executor = DockerPolicyExecutor(settings=runtime.coding.sandbox)
        action_ids = {
            option.option_id: index + 100
            for index, option in enumerate(library.manifest.options)
            if option.controller is not None
        }
        execution_records: dict[str, dict[str, Any]] = {}
        evaluation_execution: dict[str, Any] = {}
        for problem in (practice, evaluation):
            assert isinstance(problem.env, AgenticTossing3DEnvironment)
            for option in library.manifest.options:
                if option.controller is not None:
                    problem.env.register_policy(
                        skill_id=action_ids[option.option_id],
                        name=option.option_id,
                        executor=OptionExecutor(
                            env=problem.env,
                            option=option,
                            executor=executor,
                            learner=learner,
                            records=execution_records if problem is practice else {},
                            evaluation_judge=judge if problem is evaluation else None,
                            audit_path=output / "evaluation_option_judgments.jsonl",
                            observation_mode=observation_mode,
                            last_execution=evaluation_execution if problem is evaluation else None,
                        ),
                    )
        belief_backend = TossingLearnerBelief(
            linear_cost_lambda=config.linear_cost_lambda, seed=config.seed
        )
        assert isinstance(practice.env, AgenticTossing3DEnvironment)
        assert isinstance(evaluation.env, AgenticTossing3DEnvironment)
        method = AgenticOptionsMethod(
            env=practice.env,
            evaluation_env=evaluation.env,
            practice_observer=practice_observer,
            evaluation_observer=evaluation_observer,
            evaluation_skill_order=tuple(bundle["evaluation_skill_order"]),
            judge=judge,
            learner=learner,
            learner_belief=belief_backend,
            chain=SkillChainModel(manifest=library.manifest, learner_belief=belief_backend),
            belief=make_default_tossing3d_belief(
                seed=config.seed,
                num_particles=config.num_particles,
                model="local_trend",
                additional_skill_names=tuple(
                    option.belief_skill for option in library.manifest.options
                ),
            ),
            config=config,
            action_ids=action_ids,
            human_destinations={
                option.option_id: bundle["human_interventions"][option.option_id]
                for option in library.manifest.options
                if option.human_destination is not None
            },
            audit_path=output / "agentic_judgments.jsonl",
            execution_records=execution_records,
        )
        # Pydantic validates/copies dict fields during construction; callbacks and
        # the method must share the same live completion journal afterwards.
        method.execution_records = execution_records
        method.evaluation_execution = evaluation_execution
        MethodRunner.run(
            args=args,
            method=method,
            problem=practice,
            evaluation_problem=evaluation,
            practice_initialized=True,
            num_cycles=args.num_cycles,
            max_steps_per_interaction=args.max_steps_per_interaction,
            renderer=Tossing3DRenderer,
            render_fps=10,
        )

    @staticmethod
    def adapt_problem(*, problem: Tossing3DProblem) -> Tossing3DProblem:
        env = AgenticTossing3DEnvironment(**problem.env.model_dump())
        return problem.model_copy(
            update={
                "env": env,
                "tasks": problem.tasks.model_copy(update={"env": env}),
            }
        )

    @staticmethod
    def configure_library(
        *,
        library: GeneratedLibrary,
        bundle: dict[str, Any],
        human_enabled: bool,
        human_cost: float,
    ) -> GeneratedLibrary:
        robot_skills = {bundle["belief_skills"][name] for name in ("pick", "toss", "open_gripper")}
        robot_options = bundle["robot_options"]
        options = []
        for option in library.manifest.options:
            if option.controller is not None:
                if (
                    robot_options.get(option.option_id) != option.belief_skill
                    or option.option_id in bundle["human_interventions"]
                ):
                    raise ValueError("A generated option cannot change its configured belief role")
                cost = bundle["configured_costs"]["robot"]
            else:
                if not human_enabled:
                    continue
                if option.option_id not in bundle["human_interventions"]:
                    raise ValueError("A generated option cannot invent a human intervention")
                if option.belief_skill != bundle["belief_skills"]["human_reset"]:
                    raise ValueError("Human intervention must use its configured cost/belief slot")
                cost = human_cost
            options.append(OptionContract.model_validate({**option.model_dump(), "cost": cost}))
        if {item.belief_skill for item in options if item.controller} != robot_skills:
            raise ValueError("The canonical deployment model requires pick, toss and open-gripper")
        if {item.option_id for item in options if item.controller} != set(robot_options):
            raise ValueError("Generated robot option IDs must match the configured library")
        allowed = {item.option_id for item in options}
        value = library.model_dump()
        value["manifest"]["options"] = [item.model_dump() for item in options]
        value["manifest"]["edges"] = [
            edge.model_dump() for edge in library.manifest.edges if edge.option_id in allowed
        ]
        return GeneratedLibrary.model_validate(value)


class OptionExecutor:
    """Trusted host callback; generated Python is passed only to the sandbox."""

    def __init__(
        self,
        *,
        env: AgenticTossing3DEnvironment,
        option: OptionContract,
        executor: DockerPolicyExecutor,
        learner: CodePolicyLearner,
        records: dict[str, dict[str, Any]],
        evaluation_judge: VLMJudge | None,
        audit_path: Path,
        observation_mode: ObservationMode = "rgb",
        last_execution: dict[str, Any] | None = None,
    ) -> None:
        self.env, self.option, self.executor, self.learner = env, option, executor, learner
        self.records = records
        self.observation_mode = observation_mode
        self.last_execution = last_execution
        self.evaluation_judge, self.audit_path = evaluation_judge, audit_path

    def __call__(self, *, params: np.ndarray) -> ControllerRun:
        del params
        bridge = Tossing3DAgenticBridge(
            env=self.env,
            step_limit=self.option.max_steps,
            observation_mode=self.observation_mode,
        )
        result = self.executor.execute(
            library=self.learner.library, option=self.option, bridge=bridge
        )
        self.records[self.option.option_id] = result.model_dump(exclude={"final_observation"})
        if self.last_execution is not None:
            self.last_execution.clear()
            self.last_execution.update(self.records[self.option.option_id])
        if self.evaluation_judge is not None:
            judgments: dict[str, Any] = {
                "option_id": self.option.option_id,
                "execution": self.records[self.option.option_id],
            }
            try:
                observation = ObservationEvidence.with_history(
                    observation=result.final_observation, trajectory_path=result.trajectory_path
                )
                observation["execution"] = self.records[self.option.option_id]
                for phase in ("termination", "success"):
                    judgments[phase] = self.evaluation_judge.check(
                        observation=observation, option=self.option, phase=phase
                    ).model_dump()
            except Exception as exc:
                judgments["judgment_error"] = type(exc).__name__
            with self.audit_path.open("a") as stream:
                stream.write(json.dumps(judgments) + "\n")
        return ControllerRun(
            steps=result.steps, terminated=result.controller_done, error=result.error
        )
