"""Single-seed full-agent pilot: live persistent practice and frozen evaluation."""

import argparse
import json
import multiprocessing
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from hitl_pmp.agentic_runtime.docker_cli import RobocodeDockerTransport
from hitl_pmp.agentic_runtime.sandbox import (
    DockerContainer,
    DockerPolicyExecutor,
    RobotRelay,
    SandboxSettings,
)
from hitl_pmp.core.practice_costs import PracticeCosts
from hitl_pmp.environments.tossing3d.agentic_bridge import Tossing3DAgenticBridge
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks
from hitl_pmp.step_protocol import EvaluationClock, EvaluationStopping

from .world import Files, Server, World

HERE = Path(__file__).resolve().parent


class EvaluationExecutor(DockerPolicyExecutor):
    def command(self, *, work_dir: Path, name: str) -> list[str]:
        return self.container.command(
            name=name,
            work_dir=work_dir,
            argv=[
                "/sandbox/transport.py",
                "/sandbox/transport.json",
                self.settings.container_python,
                "/sandbox/evaluation_worker.py",
            ],
        )


class ScoreBridge:
    def __init__(self, *, bridge: Any, budget: int) -> None:
        self.bridge = bridge
        self.damage_events: list[dict[str, Any]] = []
        self.clock = EvaluationClock(budget=budget, check_goal=bridge.env.backend().check_goals)

    @property
    def success(self) -> bool:
        return self.clock.solved

    def observe(self) -> dict[str, Any]:
        return self.bridge.observe()

    def action_spec(self) -> dict[str, Any]:
        return self.bridge.action_spec()

    def step(self, *, action: Any) -> dict[str, Any]:
        if self.clock.solved or self.clock.steps >= self.clock.budget:
            raise RuntimeError("episode_finished")
        observation = self.bridge.step(action=action)
        self.damage_events.extend(getattr(self.bridge, "damage_events", lambda: [])())
        self.clock.after()
        return observation


class DeadlineTransport(RobocodeDockerTransport):
    deadline = float("inf")

    def run(self, **kwargs: Any) -> Any:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Global adaptation deadline reached")
        self.timeout_seconds = remaining if remaining != float("inf") else None
        return super().run(**kwargs)


class FullAgenticRunner:
    @staticmethod
    def validate_runtime(*, settings: SandboxSettings) -> None:
        """Fail closed if an operator selects the old robot-planning image."""
        container = DockerContainer(settings=settings)
        name = f"hitl-full-preflight-{time.time_ns()}"
        with tempfile.TemporaryDirectory(prefix="hitl-runtime-check-") as temporary:
            command = container.command(
                name=name,
                work_dir=Path(temporary),
                argv=[
                    "-c",
                    "import importlib.util,pathlib; "
                    "assert not pathlib.Path('/opt/hitl-planning').exists(), "
                    "'Planning assets exposed'; "
                    "assert all(importlib.util.find_spec(n) is None for n in "
                    "['pybullet','mujoco','kinder','kinder_models']), "
                    "'Simulator or planning module exposed'",
                ],
            )
            try:
                subprocess.run(command, check=True, timeout=30, capture_output=True)
            finally:
                container.remove(name=name)

    @staticmethod
    def evaluate(
        *,
        submission: Path,
        settings: SandboxSettings,
        output: Path,
        count: int,
        budget: int = 500,
        seed: int = 0,
        damage_cost: float | None = None,
        mat_size: float = 4,
    ) -> list[dict[str, Any]]:
        output.mkdir(parents=True, exist_ok=True)
        if not (submission / "approach.py").is_file():
            records = [
                dict(task=i, solved=False, steps=0, error="missing_controller")
                for i in range(count)
            ]
            Files.atomic_json(
                path=output / "results.json",
                value=dict(tasks=records, num_solved=0, num_total=count, complete=True),
            )
            return records
        env = Tossing3DEnvironment(damage_cost=damage_cost, mat_size=mat_size)
        tasks = Tossing3DTasks(env=env, seed=seed)
        results = []
        try:
            for index in range(count):
                task = tasks.sample_test_task()
                bridge = ScoreBridge(
                    bridge=Tossing3DAgenticBridge(env=env, observation_mode="object_state"),
                    budget=budget,
                )
                task_seed = int(task.initial_state.get(obj=env.scene, feature_name="seed"))
                with tempfile.TemporaryDirectory(prefix="hitl-full-eval-") as temporary:
                    work = Path(temporary)
                    shutil.copytree(submission, work / "submission")
                    shutil.copyfile(settings.transport_source(), work / "transport.py")
                    shutil.copyfile(
                        HERE.parent / "agentic_runtime/policy_worker.py", work / "policy_worker.py"
                    )
                    shutil.copyfile(
                        Path(__file__).with_name("evaluation_worker.py"),
                        work / "evaluation_worker.py",
                    )
                    Files.atomic_json(
                        path=work / "transport.json", value=dict(listeners=[], strict_blackbox=True)
                    )
                    Files.atomic_json(
                        path=work / "evaluation_config.json", value=dict(max_steps=budget)
                    )
                    relay = RobotRelay(
                        path=work / "robot.sock",
                        bridge=bridge,
                        max_steps=budget,
                        trajectory_path=output / f"task-{index:02d}.jsonl",
                    )
                    try:
                        error = EvaluationExecutor(settings=settings)._run(work=work, relay=relay)
                        error = error or relay.error
                        if not relay.finished:
                            error = error or "Policy failed to finish"
                        if bridge.success or bridge.clock.steps >= budget:
                            error = None
                        results.append(
                            dict(
                                task=index,
                                seed=task_seed,
                                solved=bool(bridge.success),
                                steps=bridge.clock.steps,
                                damage_contacts=len(bridge.damage_events),
                                damage_cost=sum(e["cost"] for e in bridge.damage_events),
                                error=error,
                            )
                        )
                        Files.atomic_json(
                            path=output / "results.json",
                            value=dict(
                                tasks=results,
                                num_solved=sum(r["solved"] for r in results),
                                num_total=len(results),
                                complete=len(results) == count,
                            ),
                        )
                        print(
                            f"Evaluation {output.name}: {len(results)}/{count} tasks complete",
                            flush=True,
                        )
                    finally:
                        relay.server_close()
        finally:
            env.close()
        return results

    @staticmethod
    def task_prompt(*, costs: PracticeCosts) -> str:
        prompt = (HERE / "task_prompt.md").read_text()
        if costs == PracticeCosts():
            return prompt
        start = prompt.index("Initial function values and fixed weights:")
        end = prompt.index("w_H weights human cost", start)
        description = (
            "Configured cost and duration functions "
            "(constant value unless a host function is named):\n```json\n"
            + costs.model_dump_json(indent=2)
            + "\n```\nThe host returns current robot-step prices "
            "and per-human-skill price/duration quotes.\n"
        )
        prompt = prompt[:start] + description + prompt[end:]
        return prompt.replace(
            "both equal to 1 initially", "specified in the configuration below"
        ).replace(
            "with dₕ(e) = 1 for each human skill",
            "using the configured duration of each human skill",
        )

    @staticmethod
    def variant_prompt(
        *, prompt: str, variant: str, fragile: bool, mat_size: float, damage_cost: float
    ) -> str:
        if variant not in {"original", "new-wording"}:
            raise ValueError("Unknown prompt variant")
        if variant == "new-wording":
            prompt += "\n" + (HERE / "continuation_cost_prompt.md").read_text()
        if fragile:
            prompt = prompt.replace("C = C_R + w_H C_H", "C = C_R + w_H C_H + C_D")
            prompt += (
                "\nThe cube is fragile. A protective squishy square mat of side "
                f"{mat_size:g} m surrounds the bin, centered on and following "
                "its x/y position and yaw. "
                "The mat can overlap the wall and robot footprint. "
                "It has no collision geometry "
                "and does not change contact physics; its protection is modeled through cost. "
                f"Each new cube impact on bare ground outside the mat costs {damage_cost:g} "
                "under the configured damage_contact function. C_D is the sum of these damage "
                "charges and is included in total practice cost C. "
                "All robot steps and human "
                "requests retain their configured costs. Contacts inside the mat, with the bin, "
                "or with the wall add no damage charge. Initial placement and human placement "
                "are exempt. Resting contact does not repeatedly charge; separate rebounds can. "
                "Damage does not add practice steps. Deployment success still requires the cube "
                "inside the bin. The host reports damage events and accumulated damage cost. "
                "Include expected future damage alongside robot and human costs "
                "in your decisions.\n"
            )
        return prompt

    @staticmethod
    def initial_files(*, workspace: Any, bridge: Any) -> None:
        shutil.copyfile(HERE / "env_client.py", workspace / "env_client.py")
        Files.atomic_json(path=workspace / "initial_observation.json", value=bridge.observe())
        shutil.copyfile(HERE / "robot_api.md", workspace / "robot_api.md")

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--sandbox-settings", type=Path, required=True)
        parser.add_argument("--practice-step-budget", type=int, default=85000)
        parser.add_argument("--measurement-interval-steps", type=int, default=1700)
        parser.add_argument("--human-skill-steps", type=int, default=1)
        parser.add_argument("--stop-after-perfect-evaluations", type=int, default=0)
        parser.add_argument("--practice-cost-config", type=Path, default=None)
        parser.add_argument("--evaluation-control-steps", type=int, default=500)
        parser.add_argument("--num-test-tasks", type=int, default=10)
        parser.add_argument("--fragile-object", action="store_true")
        parser.add_argument("--mat-size", type=float, default=4)
        parser.add_argument(
            "--prompt-variant", choices=("original", "new-wording"), default="original"
        )
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--hours", type=float, default=None)
        parser.add_argument("--model-budget", type=float, default=20)
        args = parser.parse_args()
        if any(
            value <= 0
            for value in (
                args.practice_step_budget,
                args.measurement_interval_steps,
                args.human_skill_steps,
                args.evaluation_control_steps,
                args.num_test_tasks,
                args.model_budget,
            )
        ):
            raise ValueError("Budgets and intervals must be positive")
        costs = PracticeCosts.load(path=args.practice_cost_config)
        if args.human_skill_steps != 1:
            raise ValueError("Configure per-skill durations in --practice-cost-config")
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        workspace = output / "coding" / "sandbox"
        workspace.mkdir(parents=True)
        protocol = dict(
            method="full-agent",
            environment="FragileTossing3D" if args.fragile_object else "Tossing3D",
            mat_size=args.mat_size,
            prompt_variant=args.prompt_variant,
            status="step_matched",
            seed=args.seed,
            adaptation_hours=args.hours,
            max_model_cost_usd=args.model_budget,
            cost_functions=costs.model_dump(mode="json"),
            max_trials=None,
            max_control_steps=args.practice_step_budget,
            max_human_requests=None,
            objective=(
                "E[autonomous_success] - lambda * "
                "(sum_robot_step_cost + human_weight * sum_human_skill_cost + sum_damage_cost)"
            ),
            evaluation=dict(
                num_tasks=args.num_test_tasks,
                control_steps=args.evaluation_control_steps,
                interval=args.measurement_interval_steps,
                asynchronous=True,
            ),
            human_skill_steps=args.human_skill_steps,
            stop_after_perfect_evaluations=args.stop_after_perfect_evaluations,
            practice_reset="initialization only; subsequent repositioning via charged human tools",
            learning_schedule="agent chooses updates; measurement does not trigger learning",
        )
        Files.atomic_json(path=output / "protocol.json", value=protocol)
        settings = SandboxSettings.model_validate(json.loads(args.sandbox_settings.read_text()))
        FullAgenticRunner.validate_runtime(settings=settings)
        sys.path.insert(0, str(settings.robocode_checkout.resolve() / "src"))
        settings = settings.model_copy(
            update={"artifact_dir": output / "execution", "timeout_seconds": 60}
        )
        env = Tossing3DEnvironment(
            damage_cost=costs.damage_contact.value if args.fragile_object else None,
            mat_size=args.mat_size,
        )
        pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        futures = []
        socket_dir = Path(tempfile.mkdtemp(prefix="hitl-full-sock-", dir="/tmp"))
        early_stop = False
        coder_error = []
        coder_result = []
        world = None
        thread = None
        try:
            env.hard_reset()
            bridge = Tossing3DAgenticBridge(
                env=env, observation_mode="object_state", step_limit=2**63 - 1
            )
            FullAgenticRunner.initial_files(workspace=workspace, bridge=bridge)
            Files.snapshot(
                workspace=workspace,
                destination=output / "initial_submission",
                require_controller=False,
            )
            deadline = time.monotonic() + args.hours * 3600 if args.hours else float("inf")
            world = World(bridge=bridge, workspace=workspace, output=output, deadline=deadline)

            def queue_measurement(record: Any) -> Any:  # noqa: PLR0917 -- callback
                future = pool.submit(
                    FullAgenticRunner.evaluate,
                    submission=Path(record["snapshot"]),
                    settings=settings,
                    output=output / "evaluations" / f"{record['index']:04d}",
                    count=args.num_test_tasks,
                    budget=args.evaluation_control_steps,
                    seed=args.seed,
                    damage_cost=env.damage_cost,
                    mat_size=env.mat_size,
                )
                futures.append((record, future))

            world.configure_measurements(
                budget=args.practice_step_budget,
                interval=args.measurement_interval_steps,
                human_steps=args.human_skill_steps,
                callback=queue_measurement,
                costs=costs,
            )
            prompt = FullAgenticRunner.task_prompt(costs=costs)
            prompt = FullAgenticRunner.variant_prompt(
                prompt=prompt,
                variant=args.prompt_variant,
                fragile=args.fragile_object,
                mat_size=args.mat_size,
                damage_cost=costs.damage_contact.value,
            )
            prompt += (
                f"\nPractice has a total budget of {args.practice_step_budget} counted steps: "
                "one per robot control period plus each human skill's configured duration. "
                "The host takes frozen deployment snapshots every "
                f"{args.measurement_interval_steps} "
                "steps without resetting the world "
                "or requesting a learning update. "
                "Evaluation results are never returned. "
                "Stop gracefully when the remaining budget is zero.\n"
            )
            system = (HERE / "system_prompt.md").read_text()
            (output / "task_prompt.md").write_text(prompt)
            (output / "system_prompt.md").write_text(system)
            coding_settings = settings.model_copy(update={"live_tools_dir": socket_dir})
            transport = DeadlineTransport(
                sandbox=coding_settings, backend="claude", subagent_policy="legacy"
            )
            transport.deadline = deadline
            launcher, backends, omega = transport.modules()
            config = launcher.SandboxConfig(
                sandbox_dir=workspace,
                init_files={},
                prompt=prompt,
                system_prompt=system,
                output_filename="approach.py",
                model="claude-opus-5-5",
                effort="high",
                max_budget_usd=args.model_budget,
                max_turns=0,
                max_output_tokens=32768,
                autocompact_pct=80,
                blackbox=True,
                mcp_tools=(),
            )
            backend = backends.create_backend(
                omega.DictConfig(dict(backend="claude", model="claude-opus-5-5", effort="high"))
            )

            def code() -> None:
                try:
                    result = transport.run_with_recovery(
                        config=config, backend=backend, launcher=launcher
                    )
                    coder_result.append(result)
                    Files.atomic_json(
                        path=output / "coding_result.json",
                        value=dict(
                            success=result.success,
                            error=result.error,
                            total_cost_usd=result.total_cost_usd,
                            generation_metrics=(
                                vars(result.generation_metrics)
                                if result.generation_metrics
                                else None
                            ),
                        ),
                    )
                except BaseException:
                    coder_error.append(traceback.format_exc())
                    (output / "coding_error.txt").write_text(coder_error[-1])

            with Server(path=socket_dir / "world.sock", world=world) as server:
                thread = threading.Thread(target=code, daemon=True)
                thread.start()
                last_status = time.monotonic()
                while thread.is_alive():
                    server.handle_request()
                    if time.monotonic() - last_status > 30:
                        world.status(phase="adapting")
                        last_status = time.monotonic()
                    early_stop = EvaluationStopping.reached(
                        records=EvaluationStopping.completed(futures=futures),
                        patience=args.stop_after_perfect_evaluations,
                    )
                    if (
                        early_stop
                        or time.monotonic() >= deadline
                        or (output / "STOP").exists()
                        or world.uncertain
                        or world.counted_steps >= args.practice_step_budget
                    ):
                        world.deadline = 0
                        transport.deadline = 0
                        FullAgenticRunner.cleanup_container(output=output, settings=settings)
                        break
                thread.join(timeout=5)
            if world.submission is None:
                world.submission = output / "submission"
                Files.snapshot(
                    workspace=workspace, destination=world.submission, require_controller=False
                )
            endpoint = (
                "three_perfect_evaluations"
                if early_stop
                else (
                    "practice_step_budget"
                    if world.counted_steps >= args.practice_step_budget
                    else (
                        "agent_submission"
                        if world.finished
                        else (
                            "unresolved_execution"
                            if world.uncertain
                            else (
                                "external_stop"
                                if (output / "STOP").exists()
                                else (
                                    "wall_time_limit"
                                    if time.monotonic() >= deadline
                                    else (
                                        "infrastructure_failure"
                                        if coder_error
                                        else "model_budget_or_cli_end"
                                    )
                                )
                            )
                        )
                    )
                )
            )
            Files.atomic_json(
                path=output / "adaptation_endpoint.json",
                value=dict(
                    endpoint=endpoint,
                    agent_finished=world.finished,
                    coder_exception=bool(coder_error),
                    coding_success=coder_result[0].success if coder_result else None,
                ),
            )
            world.final_measurement()
            world.status(phase="awaiting_evaluations", endpoint=endpoint)
            pool.shutdown(wait=True)
            results = []
            for record, future in futures:
                try:
                    tasks = future.result()
                    results.append(
                        dict(
                            **record,
                            num_solved=sum(t["solved"] for t in tasks),
                            num_total=len(tasks),
                            tasks=tasks,
                        )
                    )
                except Exception:
                    results.append(dict(**record, error=traceback.format_exc()))
            Files.atomic_json(path=output / "step_evaluations.json", value=results)
            world.status(
                phase="evaluation_failed" if any("error" in r for r in results) else "complete",
                endpoint=endpoint,
                evaluations_completed=len(results),
            )
        except BaseException:
            (output / "failure.txt").write_text(traceback.format_exc())
            if world:
                world.status(phase="failed")
            else:
                Files.atomic_json(
                    path=output / "status.json",
                    value=dict(phase="failed", updated_unix=time.time()),
                )
            raise
        finally:
            FullAgenticRunner.cleanup_container(output=output, settings=settings)
            pool.shutdown(wait=True)
            if world:
                world.db.close()
            env.close()
            shutil.rmtree(socket_dir, ignore_errors=True)

    @staticmethod
    def cleanup_container(*, output: Any, settings: Any) -> None:
        command_path = output / "coding" / "container_command.json"
        if command_path.exists():
            command = json.loads(command_path.read_text())
            name = command[command.index("--name") + 1]
            DockerContainer(settings=settings).remove(name=name)


if __name__ == "__main__":
    FullAgenticRunner.main()
