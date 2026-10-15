"""A fake coding agent that writes fixed modules, and a LightSwitch method wired to it."""

import json
from pathlib import Path

from hitl_pmp.core.method.types import ParameterSpec, Skill
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.methods.practice_makes_perfect.coded_sampler.coder import Coder
from hitl_pmp.methods.practice_makes_perfect.coded_sampler.history import TrajectoryHistory
from hitl_pmp.methods.practice_makes_perfect.coded_sampler.runner import SubprocessCodeRunner
from hitl_pmp.methods.practice_makes_perfect.coded_sampler.sampler import CodedSamplerHub
from hitl_pmp.methods.practice_makes_perfect.coded_sampler.types import (
    CodedSamplerMode,
    CoderResult,
)
from hitl_pmp.methods.practice_makes_perfect.coded_sampler.workspace import CodingWorkspace
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod

# Scores each row by its last entry (the parameter), so the argmax is the largest.
SCORER = "def score(sampler_inputs):\n    return [float(row[-1]) for row in sampler_inputs]\n"
SAMPLER = "def sample(state, grounding, rng):\n    return [0.25]\n"


class FakeCoder(Coder):
    """Writes `sources[skill]` into every module and records calls. Reports cost the way
    the Claude CLI does: `total_cost_usd` is cumulative over the *session*, so a resumed
    session's figure includes every earlier invocation.

    `costs[i]` is what invocation i itself spends (default 0). An invocation in
    `new_session_at` starts a fresh session, whose cumulative restarts at 0. One in
    `missing_cost` reports no cost. One in `cut_off` writes `cut_off_source` and reports
    the CLI's budget stop."""

    workspace: Path
    sources: dict[str, str]
    calls: list[dict[str, object]] = []
    costs: list[float] = []
    new_session_at: set[int] = set()
    missing_cost: set[int] = set()
    cut_off: set[int] = set()
    cut_off_source: str = "def score(sampler_inputs):\n    return [0.0] * len(sampler_inputs)\n"

    def run(
        self, *, prompt: str, system_prompt: str, resume: bool, max_budget_usd: float = 0.0
    ) -> CoderResult:
        index = len(self.calls)
        self.calls.append({"prompt": prompt, "resume": resume, "max_budget_usd": max_budget_usd})
        samplers = self.workspace / "samplers"
        samplers.mkdir(parents=True, exist_ok=True)
        cut = index in self.cut_off
        for skill, source in self.sources.items():
            (samplers / f"{skill}.py").write_text(self.cut_off_source if cut else source)
        (self.workspace.parent / "fake_session.json").write_text(
            json.dumps({"seen_data": (self.workspace / "data" / "trajectories.jsonl").exists()})
        )
        sessions = 1 + sum(1 for i in self.new_session_at if i <= index)
        session = f"fake-session-{sessions}"
        first = max([0, *(i for i in self.new_session_at if i <= index)])
        cumulative = sum(self.costs[first : index + 1])
        if index in self.missing_cost:
            return CoderResult(success=True, session_id=session, resumed=resume)
        return CoderResult(
            success=not cut,
            error="error_max_budget_usd" if cut else None,
            session_id=session,
            resumed=resume,
            session_cumulative_usd=cumulative,
            session_costs={session: cumulative},
            budget_exhausted=cut,
        )


class DescribedLightSwitchProvider(LightSwitchSkillProvider):
    """LightSwitch with its one-dimensional parameters described, for sampler mode."""

    def parameter_specs(self, *, skill: Skill) -> tuple[ParameterSpec, ...] | None:
        if skill.param_dim == 0:
            return None
        return (ParameterSpec(name="dlight", unit="cells", meaning="m", lower=-1.0, upper=1.0),)


class CodedLightSwitch:
    """A static-method container, never instantiated."""

    @staticmethod
    def build(
        *,
        tmp_path: Path,
        mode: CodedSamplerMode,
        sources: dict[str, str] | None = None,
        epsilon: float = 0.5,
        model_budget_usd: float = 0.0,
    ) -> tuple[EesMethod, LightSwitchEnvironment, CodedSamplerHub, FakeCoder]:
        env = LightSwitchEnvironment(grid_size=4)
        provider = DescribedLightSwitchProvider(env=env)
        learned = [skill.name for skill in provider.skills() if skill.param_dim > 0]
        default = SCORER if mode == CodedSamplerMode.SCORER else SAMPLER
        sandbox = tmp_path / "coding" / "sandbox"
        sandbox.mkdir(parents=True)
        coder = FakeCoder(
            workspace=sandbox,
            sources=sources if sources is not None else dict.fromkeys(learned, default),
            calls=[],
        )
        hub = CodedSamplerHub(
            mode=mode,
            provider=provider,
            workspace=CodingWorkspace(root=sandbox, mode=mode),
            coder=coder,
            runner=SubprocessCodeRunner(timeout_seconds=30.0),
            output_dir=tmp_path,
            history=TrajectoryHistory(),
            model_budget_usd=model_budget_usd,
        )
        method = EesMethod(
            env=env,
            skill_provider=provider,
            seed=0,
            coded_sampler=hub,
            exploration_epsilon=epsilon,
        )
        return method, env, hub, coder
