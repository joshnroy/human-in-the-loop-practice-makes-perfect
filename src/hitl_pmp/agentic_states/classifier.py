"""Logged language-state classification of the same object observation as coding."""

import json
import sys
import uuid
from pathlib import Path

from hitl_pmp.agentic_runtime.types import ClusterJudgment
from hitl_pmp.full_agentic.runner import DeadlineTransport
from hitl_pmp.hybrid_skills.artifacts import ModelBudgetExhausted
from hitl_pmp.step_protocol import StepFiles

from .budget import SharedBudget


class LanguageClassifier:
    def __init__(
        self, *, settings, output: Path, budget_path: Path, budget_limit: float, phase: str
    ):
        self.settings, self.output, self.phase = settings, output, phase
        self.budget_path, self.budget_limit = budget_path, budget_limit
        self.last_key = None
        self.last_result = None

    def classify(self, *, manifest, observation):
        key = (manifest.model_dump_json(), json.dumps(observation, sort_keys=True))
        if key == self.last_key:
            return self.last_result
        call_id = uuid.uuid4().hex
        directory = self.output / self.phase / call_id
        work = directory / "sandbox"
        work.mkdir(parents=True)
        StepFiles.json(path=work / "observation.json", value=observation)
        StepFiles.json(path=work / "states.json", value=[c.model_dump() for c in manifest.clusters])
        budget = SharedBudget(
            path=self.budget_path,
            limit=self.budget_limit,
            session=f"classifier-{self.phase}-{call_id}",
        )
        sys.path.insert(0, str(self.settings.robocode_checkout.resolve() / "src"))
        transport = DeadlineTransport(
            sandbox=self.settings, backend="claude", subagent_policy="disallowed"
        )
        launcher, backends, omega = transport.modules()
        with budget.lock():
            if budget.exhausted:
                raise ModelBudgetExhausted()
            config = launcher.SandboxConfig(
                sandbox_dir=work,
                init_files={},
                prompt=(
                    "Read observation.json and states.json. Classify this observation into "
                    "exactly one"
                    "language-defined state. If none fits or membership is ambiguous, use null. "
                    "Write judgment.json with exactly cluster_id (string or null) and "
                    "reason (a short"
                    "explanation citing observed measurements). Do not infer success from "
                    "the requested"
                    "action or intention. Do not edit the input files or create controllers."
                ),
                system_prompt=(
                    "You classify observable robot scenes using supplied natural-language "
                    "regions. Return only the requested judgment artifact. Do not use other "
                    "agents."
                ),
                output_filename="judgment.json",
                model="claude-opus-5-5",
                effort="low",
                max_budget_usd=budget.remaining,
                max_turns=0,
                max_output_tokens=4096,
                autocompact_pct=80,
                blackbox=True,
                mcp_tools=(),
                resume_previous_session=False,
            )
            backend = backends.create_backend(
                omega.DictConfig(dict(backend="claude", model="claude-opus-5-5", effort="low"))
            )
            result = transport.run_with_recovery(config=config, backend=backend, launcher=launcher)
            budget.record(cumulative=result.total_cost_usd)
            StepFiles.json(
                path=directory / "result.json",
                value=dict(
                    success=result.success,
                    error=result.error,
                    cost_usd=result.total_cost_usd,
                    manifest_revision=manifest.revision,
                    phase=self.phase,
                ),
            )
            if not result.success:
                if budget.exhausted:
                    raise ModelBudgetExhausted()
                raise RuntimeError(f"State classification failed: {result.error}")
        judgment = ClusterJudgment.model_validate_json((work / "judgment.json").read_text())
        if judgment.cluster_id is not None and judgment.cluster_id not in {
            c.cluster_id for c in manifest.clusters
        }:
            raise ValueError("Classifier returned an unknown state ID")
        self.last_key, self.last_result = key, judgment
        return judgment
