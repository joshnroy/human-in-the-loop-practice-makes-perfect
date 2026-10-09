"""Frozen robot code executes through the existing isolated low-level bridge."""

import pickle
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from hitl_pmp.agentic_runtime.sandbox import DockerPolicyExecutor, RobotRelay, SandboxSettings
from hitl_pmp.environments.tossing3d.agentic_bridge import Tossing3DAgenticBridge
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.kinder_backend import ControllerRun
from hitl_pmp.step_protocol import DeploymentSnapshot, StepEvaluation, StepFiles

from .artifacts import SkillBundle
from .method import HybridDeploymentMethod


class SkillExecutor(DockerPolicyExecutor):
    def command(self, *, work_dir: Path, name: str) -> list[str]:
        command = super().command(work_dir=work_dir, name=name)
        command[-1] = "/sandbox/hybrid_worker.py"
        return command

    def execute_skill(
        self, *, bundle: SkillBundle, bridge: Any, skill: str, limit: int = 500
    ) -> ControllerRun:
        self.settings.artifact_dir.mkdir(parents=True, exist_ok=True)
        trajectory = self.settings.artifact_dir / f"trajectory-{uuid.uuid4().hex}.jsonl"
        with tempfile.TemporaryDirectory(prefix="hitl-hybrid-") as tmp:
            work = Path(tmp)
            bundle.write(directory=work / "submission")
            shutil.copyfile(self.settings.transport_source(), work / "transport.py")
            source = Path(__file__).parents[1] / "agentic_runtime/policy_worker.py"
            shutil.copyfile(source, work / "policy_worker.py")
            shutil.copyfile(Path(__file__).with_name("worker.py"), work / "hybrid_worker.py")
            StepFiles.json(
                path=work / "transport.json", value=dict(listeners=[], strict_blackbox=True)
            )
            StepFiles.json(
                path=work / "skill_config.json",
                value=dict(skill=skill, limit=limit, seed=self.settings.seed),
            )
            relay = RobotRelay(
                path=work / "robot.sock", bridge=bridge, max_steps=limit, trajectory_path=trajectory
            )
            try:
                error = self._run(work=work, relay=relay)
                if not relay.started:
                    raise RuntimeError("Skill worker did not authenticate")
                error = error or relay.error
                relay.record_final(observation=bridge.observe(), error=error)
                StepFiles.event(
                    path=self.settings.artifact_dir / "executions.jsonl",
                    skill=skill,
                    trajectory=trajectory.name,
                    revision=bundle.digest,
                    steps=relay.steps,
                    controller_done=relay.done,
                    error=error,
                )
                return ControllerRun(steps=relay.steps, terminated=relay.done, error=error)
            finally:
                relay.server_close()


class HybridEnvironment(Tossing3DEnvironment):
    executor: Any = Field(default=None, exclude=True)
    bundle: SkillBundle | None = None

    def _execute(self, *, action: np.ndarray) -> list[ControllerRun]:
        names = {
            self.pick_cube_id: "PickCube",
            self.move_to_toss_location_and_toss_id: "MoveToTossLocationAndToss",
            self.open_gripper_id: "OpenGripper",
        }
        name = names.get(int(action[0]))
        if name is None:
            if int(action[0]) != self.noop_id:
                raise ValueError("Hybrid cannot dispatch an unbound robot controller")
            return []
        if self.executor is None or self.bundle is None:
            raise RuntimeError("Hybrid controller has not been initialized")
        backend = self.backend()
        limit = {
            "PickCube": backend.pick_step_limit,
            "MoveToTossLocationAndToss": backend.toss_step_limit,
            "OpenGripper": backend.open_gripper_step_limit,
        }[name]
        bridge = Tossing3DAgenticBridge(env=self, observation_mode="object_state", step_limit=limit)
        return [
            self.executor.execute_skill(bundle=self.bundle, bridge=bridge, skill=name, limit=limit)
        ]


class HybridSnapshot:
    @staticmethod
    def encode(*, method: Any) -> bytes:
        if method.env.bundle is None:
            raise RuntimeError("No accepted skill code to evaluate")
        return pickle.dumps(
            dict(
                base=DeploymentSnapshot.encode(method=method), bundle=method.env.bundle.model_dump()
            ),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    @staticmethod
    def restore(*, raw: bytes, env: Any, provider: Any) -> HybridDeploymentMethod:
        payload = pickle.loads(raw)
        env.bundle = SkillBundle.model_validate(payload["bundle"])
        base = DeploymentSnapshot.restore(raw=payload["base"], env=env, provider=provider)
        method = HybridDeploymentMethod(
            env=env,
            skill_provider=provider,
            **{
                k: getattr(base, k)
                for k in type(base).model_fields
                if k not in {"env", "skill_provider"}
            },
        )
        method._competence_models = base._competence_models
        method._rng.bit_generator.state = base._rng.bit_generator.state
        return method


def prepare_problem(*, problem: Any, configuration: dict[str, Any]) -> Any:
    settings = SandboxSettings.model_validate(configuration["hybrid_sandbox"])
    env = HybridEnvironment(**problem.env.model_dump())
    env.executor = SkillExecutor(
        settings=settings.model_copy(
            update={
                "artifact_dir": Path(configuration["output_dir"])
                / "evaluation_trajectories"
                / uuid.uuid4().hex
            }
        )
    )
    problem.env = env
    problem.tasks.env = env
    return problem


class HybridEvaluation:
    @staticmethod
    def run(**kwargs: Any) -> dict[str, Any]:
        return StepEvaluation.run(
            **kwargs, prepare_problem=prepare_problem, restore_method=HybridSnapshot.restore
        )
