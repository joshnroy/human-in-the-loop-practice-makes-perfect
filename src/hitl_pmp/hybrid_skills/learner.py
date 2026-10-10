"""Persistent RoboCode skill revision, with practice-only evidence and one dollar cap."""

import json
import shutil
from pathlib import Path
from typing import Any

from hitl_pmp.agentic_runtime.sandbox import SandboxSettings
from hitl_pmp.core.practice_costs import PracticeCosts
from hitl_pmp.environments.tossing3d.agentic_bridge import Tossing3DAgenticBridge
from hitl_pmp.full_agentic.runner import HERE, DeadlineTransport, FullAgenticRunner
from hitl_pmp.step_protocol import StepFiles

from .artifacts import ModelBudgetExhausted, SessionBudget, SkillBundle


def hybrid_prompt(*, costs: PracticeCosts, mat_size: float, budget: float = 20) -> str:
    prompt = FullAgenticRunner.variant_prompt(
        prompt=FullAgenticRunner.task_prompt(costs=costs),
        variant="original",
        fragile=True,
        mat_size=mat_size,
        damage_cost=costs.damage_contact.value,
    )
    start = prompt.index("Deployment calls the interface below")
    end = prompt.index("hint:", start)
    prompt = prompt[:start] + prompt[end:]
    # Keep physical descriptions, objective and existing transfer hint verbatim.
    # Replace only the orchestration and output contract that differ in this arm.
    prompt = prompt.replace(
        "Use `env_client.py` for serialized robot commands and human requests. "
        "The deployment interface is `GeneratedApproach`; write its implementation.",
        "A fixed expectimax planner selects skills and human interventions. "
        "You implement and improve the robot skills in submission/skills.py.",
    )
    prompt = prompt.replace(
        "When finished, commit your controller and call `env.finish_adaptation()`.",
        "When this revision is ready, commit the code and end your response.",
    )
    return (
        prompt
        + """

HYBRID EXECUTION CONTRACT:
The planner owns physical practice; you own robot skill code. Code edits and offline
analysis use the same coding tools/packages as the full-agentic condition. No separate
simulator or geometry/planning service is provided. Read robot_api.md and the current
observation. Read full observation/action trajectories and accounting in evidence/.
Do not fabricate trajectories. Held-out evaluation results are unavailable.
After you return, the host freezes this revision and executes planner-selected skills
in the persistent world. It resumes this conversation with new recorded evidence at
planner STOP or after 20 skill invocations. Measurement alone does not request revision.
Do not execute physical commands or human resets outside this planner-controlled loop.
Both human skills described above remain available to the planner.

Implement GeneratedSkills(action_space, observation_space, primitives) in
submission/skills.py. The constructor gets the same action metadata, object-state
descriptor and empty primitives dictionary as full-agentic. reset(observation, skill)
starts one named skill; get_action(observation) returns a valid low-level vector or
millisecond schedule, or None to finish that invocation. No model calls occur during
execution. Helper Python files under submission/ are allowed. Skill identities:
- PickCube: acquire and hold the cube.
- MoveToTossLocationAndToss: move and deliver the held cube into the bin.
- OpenGripper: open the gripper to recover an empty usable hand.
The structured host retains its existing skill initiation/effect definitions.
Existing controller limits apply: PickCube 400 steps, toss 1000, OpenGripper 100;
the common remaining practice or evaluation budget can interrupt earlier.
Write complete code, not pseudocode. Explain the evidence and reason for each revision
briefly in your ordinary response; a no-change revision is permitted. This is a fresh
bootstrap, not a previously learned controller. There is one $20 total model budget
across initialization and all revisions, not $20 per revision.
"""
    ).replace("$20", f"${budget:g}")


class SkillLearner:
    def __init__(
        self,
        *,
        output: Path,
        env: Any,
        settings: SandboxSettings,
        costs: PracticeCosts,
        budget: float = 20,
    ) -> None:
        self.output, self.env, self.settings = output, env, settings
        self.costs = costs
        self.budget = SessionBudget(limit=budget)
        self.workspace = output / "coding/sandbox"
        self.workspace.mkdir(parents=True)
        self.submission = self.workspace / "submission"
        self.submission.mkdir()
        shutil.copyfile(Path(__file__).with_name("bootstrap.py"), self.submission / "skills.py")
        self.env.bundle = SkillBundle.read(directory=self.submission)
        self.calls = 0
        self.transport = DeadlineTransport(
            sandbox=settings, backend="claude", subagent_policy="legacy"
        )
        self.prompt = hybrid_prompt(costs=costs, mat_size=env.mat_size, budget=budget)
        (output / "task_prompt.md").write_text(self.prompt)
        # Same coding tools and generic RoboCode system prompt, with the hybrid
        # ownership contract superseding the generic fresh-world/testing guidance.
        self.system = (HERE / "system_prompt.md").read_text() + (
            "\nThe HYBRID EXECUTION CONTRACT overrides who initiates physical testing: "
            "return a revision for the host planner to practice, then inspect its evidence.\n"
        )
        (output / "system_prompt.md").write_text(self.system)

    def prepare_evidence(self) -> None:
        bridge = Tossing3DAgenticBridge(env=self.env, observation_mode="object_state")
        StepFiles.json(path=self.workspace / "current_observation.json", value=bridge.observe())
        if self.calls == 0:
            shutil.copyfile(
                self.workspace / "current_observation.json",
                self.workspace / "initial_observation.json",
            )
        api = (HERE / "robot_api.md").read_text()
        (self.workspace / "robot_api.md").write_text(api[api.index("### Robot API") :])
        evidence = self.workspace / "evidence"
        evidence.mkdir(exist_ok=True)
        trajectory_dir = self.settings.artifact_dir
        if trajectory_dir.exists():
            for source in trajectory_dir.glob("*.jsonl"):
                shutil.copyfile(source, evidence / source.name)
        events_path = self.output / "step_events.jsonl"
        events = []
        if events_path.exists():
            for line in events_path.read_text().splitlines():
                record = json.loads(line)
                if record.get("event") in {
                    "robot",
                    "human",
                    "damage",
                    "session_start",
                    "learning_update",
                }:
                    events.append(record)
        (evidence / "practice_events.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in events)
        )
        counted_steps = max((r.get("end_step", 0) for r in events), default=0)
        StepFiles.json(
            path=evidence / "accounting.json",
            value={
                "cost_functions": self.costs.model_dump(mode="json"),
                "counted_steps": counted_steps,
                "remaining_practice_steps": max(0, 85000 - counted_steps),
                "physical_cost": sum(
                    r.get("cost", 0) or 0 for r in events if r.get("event") in {"robot", "human"}
                ),
                "remaining_model_budget_usd": self.budget.remaining,
            },
        )
        # Deliberately omit status.json, evaluation directories, symbolic labels,
        # belief diagnostics and any host source/simulator handles.

    def validate_revision(self) -> None:
        """Validate additional generated artifacts before publishing controller code."""

    def revise(self) -> None:
        if self.budget.exhausted:
            raise ModelBudgetExhausted()
        self.prepare_evidence()
        launcher, backends, omega = self.transport.modules()
        prompt = (
            self.prompt
            if self.calls == 0
            else (
                "The planner completed another practice session. Read current_observation.json "
                "and new evidence/ records, revise submission/skills.py and any helpers as useful, "
                "then commit and return. The same hybrid contract and total model budget apply. "
                "Explain what you changed and why; no change is allowed."
            )
        )
        if self.calls > 0:
            prompt += getattr(self, "revision_instruction", "")
        config = launcher.SandboxConfig(
            sandbox_dir=self.workspace,
            init_files={},
            prompt=prompt,
            system_prompt=self.system,
            output_filename="submission/skills.py",
            model="claude-opus-5-5",
            effort="high",
            max_budget_usd=self.budget.remaining,
            max_turns=0,
            max_output_tokens=32768,
            autocompact_pct=80,
            blackbox=True,
            mcp_tools=(),
            resume_previous_session=self.calls > 0,
        )
        backend = backends.create_backend(
            omega.DictConfig(dict(backend="claude", model="claude-opus-5-5", effort="high"))
        )
        result = self.transport.run_with_recovery(config=config, backend=backend, launcher=launcher)
        self.budget.record(cumulative=result.total_cost_usd)
        call_dir = self.output / "coding_calls" / f"{self.calls:04d}"
        call_dir.mkdir(parents=True)
        for name in (
            "stream.jsonl",
            "broker.jsonl",
            "resume-session.json",
            "container_command.json",
        ):
            path = self.workspace.parent / name
            if path.exists():
                shutil.copyfile(path, call_dir / name)
        self.calls += 1
        bundle = SkillBundle.read(directory=self.submission)
        if not result.success:
            StepFiles.json(
                path=call_dir / "result.json",
                value=dict(
                    success=False, error=result.error, cumulative_cost_usd=self.budget.spent
                ),
            )
            if self.budget.exhausted:
                raise ModelBudgetExhausted()
            raise RuntimeError(f"Coding revision failed: {result.error}")
        self.validate_revision()
        bundle.write(directory=self.output / "revisions" / f"{self.calls:04d}")
        self.env.bundle = bundle
        StepFiles.json(
            path=call_dir / "result.json",
            value=dict(
                success=True,
                revision=bundle.digest,
                cumulative_cost_usd=self.budget.spent,
                remaining_budget_usd=self.budget.remaining,
            ),
        )
        StepFiles.json(
            path=self.output / "learner_status.json",
            value=dict(
                calls=self.calls,
                revision=bundle.digest,
                model_cost_usd=self.budget.spent,
                remaining_budget_usd=self.budget.remaining,
            ),
        )
