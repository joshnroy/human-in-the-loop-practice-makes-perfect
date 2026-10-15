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
    """Writes `sources[skill]` into every module, keeps one session id, records calls."""

    workspace: Path
    sources: dict[str, str]
    calls: list[dict[str, object]] = []

    def run(self, *, prompt: str, system_prompt: str, resume: bool) -> CoderResult:
        self.calls.append({"prompt": prompt, "resume": resume})
        samplers = self.workspace / "samplers"
        samplers.mkdir(parents=True, exist_ok=True)
        for skill, source in self.sources.items():
            (samplers / f"{skill}.py").write_text(source)
        (self.workspace.parent / "fake_session.json").write_text(
            json.dumps({"seen_data": (self.workspace / "data" / "trajectories.jsonl").exists()})
        )
        return CoderResult(success=True, session_id="fake-session", resumed=resume)


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
        )
        method = EesMethod(
            env=env,
            skill_provider=provider,
            seed=0,
            coded_sampler=hub,
            exploration_epsilon=epsilon,
        )
        return method, env, hub, coder
