"""Samplers whose decision rule is code a coding agent wrote, refitted once per cycle.

Two drop-in replacements for `LearnedSkillSampler`, sharing one `CodedSamplerHub` per
run:

- `CodedScorerSampler` *is* a `LearnedSkillSampler` whose `score_inputs` calls the
  generated `score(sampler_inputs)` instead of the MLP. The argmax wrapper, its tie
  breaking, the uninformative fallback and the epsilon-greedy override are inherited
  unchanged, RNG draws included. A failed call scores everything 0.5, which is exactly
  an unfitted MLP: a uniform draw with `was_informed=False`.
- `CodedProposalSampler` calls the generated `sample(state, grounding, rng)` for the
  greedy branch. The epsilon coin is drawn first, as `LearnedSkillSampler` draws it
  for a discriminating classifier; when it fires the call is skipped and a uniform base
  candidate is returned with `was_random=True`. A failed call or a vector that breaks
  shape or bounds is never clipped: it is logged and replaced by a uniform base
  candidate with `was_informed=False`.

Before the first refit there is no code, and both behave as an unfitted MLP does.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from hitl_pmp.core.method.skill_provider import SkillProvider
from hitl_pmp.core.method.types import GroundSkill, ParameterSpec
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.methods.practice_makes_perfect.wrapped_sampler import (
    LearnedSkillSampler,
    SamplerChoice,
)

from .coder import Coder
from .history import StateCodec, TrajectoryHistory
from .prompts import CodedSamplerPrompts
from .runner import CodeRunner
from .types import CodedSamplerMode, CodeVersion
from .workspace import CodingWorkspace, GeneratedImports, ParameterCheck


class JsonlLog(BaseModel):
    path: Path

    def write(self, *, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps(record, default=str) + "\n")


class CodedSamplerHub(BaseModel):
    """Per run: the history, the coding session, the frozen versions and every call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode: CodedSamplerMode
    provider: SkillProvider
    workspace: CodingWorkspace
    coder: Coder
    runner: CodeRunner
    output_dir: Path
    history: TrajectoryHistory

    _versions: list[CodeVersion] = PrivateAttr(default_factory=list)
    _refits: int = PrivateAttr(default=0)
    _history_cursor: int = PrivateAttr(default=0)
    _errors_since_refit: list[dict[str, Any]] = PrivateAttr(default_factory=list)
    _phase: str = PrivateAttr(default="practice")
    _decision: dict[str, Any] = PrivateAttr(default_factory=dict)
    _calls: JsonlLog = PrivateAttr()
    _refit_log: JsonlLog = PrivateAttr()
    _compactions: JsonlLog = PrivateAttr()
    _initialised: bool = PrivateAttr(default=False)

    def model_post_init(self, __context: object) -> None:
        self._versions = [CodeVersion(index=0, cycle=None, snapshot_dir=None, git_commit=None)]
        self._calls = JsonlLog(path=self.output_dir / "coded_sampler_calls.jsonl")
        self._refit_log = JsonlLog(path=self.output_dir / "coded_sampler_refits.jsonl")
        self._compactions = JsonlLog(path=self.output_dir / "compactions.jsonl")

    # ---------------------------------------------------------------- lifecycle

    @property
    def learned_skills(self) -> list[str]:
        return [skill.name for skill in self.provider.skills() if skill.param_dim > 0]

    @property
    def current_version(self) -> CodeVersion:
        return self._versions[-1]

    def set_phase(self, *, phase: str) -> None:
        self._phase = phase

    def initialise(self) -> None:
        if self._initialised:
            return
        self.workspace.initialise(
            provider=self.provider,
            instructions=CodedSamplerPrompts.instructions(
                mode=self.mode, skills=self.learned_skills, workspace=self.workspace
            ),
        )
        self._initialised = True

    def refit(self) -> None:
        """End of a practice cycle: hand the agent everything new, then freeze."""
        self.initialise()
        cycle = self.history.cycle - 1
        start = self._history_cursor
        end = len(self.history)
        data = self.workspace.root / "data"
        self.history.write(path=data / "trajectories.jsonl")
        self.history.write(path=data / f"new_since_refit_{self._refits:03d}.jsonl", start=start)
        (data / "code_errors.jsonl").write_text(
            "".join(json.dumps(error, default=str) + "\n" for error in self._errors_since_refit)
        )
        prompt = CodedSamplerPrompts.refit(
            refit=self._refits,
            cycle=cycle,
            start=start,
            end=end,
            errors=len(self._errors_since_refit),
            new_file=f"data/new_since_refit_{self._refits:03d}.jsonl",
        )
        result = self.coder.run(
            prompt=prompt,
            system_prompt=CodedSamplerPrompts.system(),
            resume=self._refits > 0,
        )
        version = self.workspace.snapshot(
            index=self._refits + 1, cycle=cycle, skills=self.learned_skills
        )
        assert version.snapshot_dir is not None
        code_dir = version.snapshot_dir / self.workspace.MODULE_DIR
        # The agent's own helpers beside its modules are importable at call time.
        local = {entry.stem for entry in code_dir.iterdir()}
        violations = {
            skill: GeneratedImports.violations(
                source=(version.snapshot_dir / module).read_text(), local_modules=local
            )
            for skill, module in version.modules.items()
        }
        rejected = {skill: names for skill, names in violations.items() if names}
        if rejected:
            version = version.model_copy(
                update={"modules": {k: v for k, v in version.modules.items() if k not in rejected}}
            )
        self._versions.append(version)
        record = {
            "refit": self._refits,
            "cycle": cycle,
            "history_events": [start, end],
            "code_errors_reported": len(self._errors_since_refit),
            "version": version.model_dump(mode="json"),
            "import_violations": rejected,
            "coder": result.model_dump(mode="json"),
        }
        self._refit_log.write(record=record)
        for index, compaction in enumerate(result.compactions):
            self._compactions.write(
                record={
                    "refit": self._refits,
                    "cycle": cycle,
                    "index_in_refit": index,
                    "session_id": result.session_id,
                    **compaction.model_dump(),
                }
            )
        # The errors just handed to the agent are reported; start the next window
        # with the new version's own import rejections, which it sees next refit.
        self._errors_since_refit = []
        for skill, names in rejected.items():
            self.record_error(skill_name=skill, error=f"disallowed imports: {names}", call=None)
        self._history_cursor = end
        self._refits += 1

    # ---------------------------------------------------------------- calls

    def begin_decision(self, *, ground_skill: GroundSkill, state: State, explore: bool) -> None:
        self._decision = {
            "skill": ground_skill.skill.name,
            "grounding": StateCodec.grounding(ground_skill=ground_skill),
            "state": StateCodec.encode(state=state),
            "explore": explore,
        }

    def has_code(self, *, skill_name: str) -> bool:
        return skill_name in self.current_version.modules

    def record_error(self, *, skill_name: str, error: str, call: dict[str, Any] | None) -> None:
        self._errors_since_refit.append({
            "skill": skill_name,
            "version": self.current_version.index,
            "phase": self._phase,
            "error": error,
            "call": call,
        })

    def log_call(self, *, record: dict[str, Any]) -> None:
        self._calls.write(
            record={
                "phase": self._phase,
                "cycle": self.history.cycle,
                "version": self.current_version.index,
                **record,
            }
        )

    def call(self, *, skill_name: str, function: str, payload: dict[str, Any]) -> Any:
        """The generated function's value, or `None` after logging why not."""
        outcome = self.runner.call(
            version=self.current_version,
            skill_name=skill_name,
            function=function,
            payload=payload,
        )
        if not outcome.ok:
            self.record_error(skill_name=skill_name, error=str(outcome.error), call=None)
        self.log_call(
            record={
                "skill": skill_name,
                "function": function,
                "ok": outcome.ok,
                "error": outcome.error,
                "latency_seconds": outcome.latency_seconds,
            }
        )
        return outcome.value if outcome.ok else None

    def decision(self) -> dict[str, Any]:
        return dict(self._decision)

    def note(self, *, record: dict[str, Any]) -> None:
        self.log_call(record={"note": True, **record})

    def parameter_specs(self, *, skill_name: str) -> list[ParameterSpec]:
        skill = next(s for s in self.provider.skills() if s.name == skill_name)
        specs = self.provider.parameter_specs(skill=skill)
        if specs is None:
            raise ValueError(f"{skill_name}: sampler mode needs the domain's parameter bounds")
        return list(specs)

    def sampler_for(
        self,
        *,
        skill_name: str,
        param_dim: int,
        num_candidates: int,
        exploration_epsilon: float,
        seed: int,
    ) -> "CodedScorerSampler | CodedProposalSampler":
        if self.mode == CodedSamplerMode.SCORER:
            return CodedScorerSampler(
                hub=self,
                skill_name=skill_name,
                param_dim=param_dim,
                num_candidates=num_candidates,
                exploration_epsilon=exploration_epsilon,
                seed=seed,
            )
        return CodedProposalSampler(
            hub=self,
            specs=self.parameter_specs(skill_name=skill_name),
            skill_name=skill_name,
            param_dim=param_dim,
            num_candidates=num_candidates,
            exploration_epsilon=exploration_epsilon,
            seed=seed,
        )


class CodedScorerSampler(LearnedSkillSampler):
    hub: CodedSamplerHub

    @property
    def is_fitted(self) -> bool:
        return self.hub.has_code(skill_name=self.skill_name)

    def fit(self) -> None:
        """Refits happen once per run, in `CodedSamplerHub.refit`."""

    def score_inputs(self, *, sampler_inputs: list[list[float]]) -> list[float]:
        if not sampler_inputs:
            raise ValueError(f"{self.skill_name}: scoring requires at least one input row.")
        unfitted = [0.5] * len(sampler_inputs)
        if not self.is_fitted:
            return unfitted
        value = self.hub.call(
            skill_name=self.skill_name,
            function="score",
            payload={"sampler_inputs": sampler_inputs},
        )
        if value is None:
            return unfitted
        if (
            not isinstance(value, list)
            or len(value) != len(sampler_inputs)
            or not all(
                isinstance(v, int | float) and not isinstance(v, bool) and np.isfinite(v)
                for v in value
            )
        ):
            error = f"score returned {str(value)[:300]!r}; expected {len(sampler_inputs)} floats"
            self.hub.record_error(skill_name=self.skill_name, error=error, call=None)
            self.hub.note(record={"skill": self.skill_name, "invalid_scores": error})
            return unfitted
        return [float(v) for v in value]


class CodedProposalSampler(BaseModel):
    """`LearnedSkillSampler`'s surface, with the greedy draw delegated to code."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hub: CodedSamplerHub
    specs: list[ParameterSpec]
    skill_name: str
    param_dim: int
    num_candidates: int = 100
    exploration_epsilon: float = 0.5
    seed: int = 0

    _rng: np.random.Generator = PrivateAttr()
    _observations: int = PrivateAttr(default=0)

    def model_post_init(self, __context: object) -> None:
        self._rng = np.random.default_rng(self.seed)

    @property
    def is_fitted(self) -> bool:
        return self.hub.has_code(skill_name=self.skill_name)

    @property
    def num_observations(self) -> int:
        return self._observations

    def observe(self, *, sampler_input: list[float], success: bool) -> None:
        """The training data is the run's history, which the hub keeps."""
        del sampler_input, success
        self._observations += 1

    def fit(self) -> None:
        """Refits happen once per run, in `CodedSamplerHub.refit`."""

    def uniform(self, *, candidates: list[np.ndarray]) -> np.ndarray:
        return candidates[int(self._rng.integers(0, len(candidates)))]

    def sample(
        self,
        *,
        sampler_inputs: list[list[float]],
        candidates: list[np.ndarray],
        explore: bool,
    ) -> SamplerChoice:
        del sampler_inputs
        if not candidates:
            raise ValueError(f"{self.skill_name}: sample requires at least one candidate.")
        if not self.is_fitted:
            return SamplerChoice(
                params=self.uniform(candidates=candidates), was_random=False, was_informed=False
            )
        if explore and self._rng.uniform() <= self.exploration_epsilon:
            return SamplerChoice(
                params=self.uniform(candidates=candidates), was_random=True, was_informed=False
            )
        decision = self.hub.decision()
        started = time.monotonic()
        value = self.hub.call(
            skill_name=self.skill_name,
            function="sample",
            payload={
                "state": decision.get("state", {}),
                "grounding": decision.get("grounding", []),
                "seed": int(self._rng.integers(0, 2**31 - 1)),
            },
        )
        if value is None:
            return SamplerChoice(
                params=self.uniform(candidates=candidates), was_random=False, was_informed=False
            )
        violation = ParameterCheck.violation(value=value, specs=self.specs)
        if violation is not None:
            self.hub.record_error(skill_name=self.skill_name, error=violation, call=None)
            self.hub.note(
                record={
                    "skill": self.skill_name,
                    "bounds_violation": violation,
                    "value": value,
                    "latency_seconds": time.monotonic() - started,
                }
            )
            return SamplerChoice(
                params=self.uniform(candidates=candidates), was_random=False, was_informed=False
            )
        return SamplerChoice(
            params=np.asarray(value, dtype=float), was_random=False, was_informed=True
        )
