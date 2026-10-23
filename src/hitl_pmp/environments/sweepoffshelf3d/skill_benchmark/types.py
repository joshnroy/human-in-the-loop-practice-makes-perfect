"""The skill benchmark's records: a stored state, the cases run from it, each case's
result, and the report a launch gate reads."""

from pydantic import BaseModel, ConfigDict, Field

# Every skill run ends in exactly one of these. The four failure labels attribute the
# failure from the attempt's per-candidate trace (`verdicts.py`):
#   no_plan          no candidate planned (KINDER's PyBullet planner, or a scripted
#                    routine's own IK/route search) -- a controller gap
#   plan_blocked     a candidate planned in PyBullet but every such plan was refused
#                    by the MuJoCo re-check -- the planning/execution simulator gap
#   execution_failed a plan ran in MuJoCo and the skill failed during or after it
#   effects_missing  the controller raised nothing, yet a promised effect is absent
# plus `budget` (cut off by the control-step limit).
FAILURE_LABELS = ("no_plan", "plan_blocked", "execution_failed", "effects_missing", "budget")


class BenchmarkReport(BaseModel):
    """What the launch gate reads: which commit and KINDER trees were benchmarked, on
    which state set, and whether every skill met its target."""

    domain: str
    git_commit: str
    git_dirty: bool
    kindergarden_commit: str
    kinder_models_commit: str
    state_set: str
    state_set_sha256: str
    scene_fingerprint: str
    passed: bool
    skills: list["SkillVerdict"]
    num_cases: int
    num_invalid: int
    elapsed_seconds: float = 0.0


class SkillVerdict(BaseModel):
    skill: str
    # "every state" | "some setting per state and target cube" | "at least x/y"
    # (| " of <families>") | "some setting for at least x/y (state, target cube) pairs"
    target: str
    passed: bool
    # Every valid run of the skill, gated or not.
    successes: int
    total: int
    # The runs the target judges, and the ones only reported (a threshold's families
    # exclude them). A skill without a threshold gates on every run.
    gated_successes: int = 0
    gated_total: int = 0
    reported_successes: int = 0
    reported_total: int = 0
    # Why the verdict failed when it is not a count below target (a changed denominator).
    reason: str = ""
    # Per simulator-attribution label, how many runs ended there.
    labels: dict[str, int] = Field(default_factory=dict)
    # Continuous-parameter skills only: per "<state>/<target cube>", settings that
    # succeeded / tried; and how many of those pairs had some setting succeed.
    per_state: dict[str, list[int]] = Field(default_factory=dict)
    gated_pairs: int = 0
    gated_pairs_total: int = 0
    # Runs whose skill planned in KINDER's PyBullet scene at every stage it reached
    # (None for a skill that does not plan there).
    pybullet_planned: int | None = None


class SkillThreshold(BaseModel):
    """A skill passes with at least `min_successes` of its gated runs succeeding, and
    only if exactly `out_of` gated runs were valid: a different count means the state
    set changed under the threshold, which is refused rather than rescaled. `families`
    are the state families that gate (None: every family); the rest are reported only."""

    model_config = ConfigDict(frozen=True)

    min_successes: int
    out_of: int
    families: frozenset[str] | None = None

    def target(self) -> str:
        line = f"at least {self.min_successes}/{self.out_of}"
        if self.families is None:
            return line
        return f"{line} of {', '.join(sorted(self.families))}"


class SettingThreshold(BaseModel):
    """A continuous-parameter skill passes when at least `min_pairs` of its (state,
    target cube) pairs have some setting that succeeds, and only if exactly `out_of`
    pairs ran: a different count means the state set changed under the threshold, which
    is refused rather than rescaled."""

    model_config = ConfigDict(frozen=True)

    min_pairs: int
    out_of: int

    def target(self) -> str:
        return (
            f"some setting for at least {self.min_pairs}/{self.out_of} (state, target cube) pairs"
        )


class CaseResult(BaseModel):
    case_id: str
    state_id: str
    family: str
    skill: str
    cube: str | None = None
    params: list[float] = Field(default_factory=list)
    # "success", one of FAILURE_LABELS, or "invalid" (the state does not satisfy the
    # skill's preconditions, so the run would measure nothing).
    label: str
    trace_outcome: str | None = None
    error: str | None = None
    # KINDER-backed skills: "planned" / "no_plan" in the PyBullet scene; "n/a" else.
    pybullet: str = "n/a"
    controller_steps: int = 0
    wall_seconds: float = 0.0
    video: str | None = None


class BenchmarkState(BaseModel):
    """One benchmark start state as fixed MuJoCo values: restoring it needs no RNG, only
    `reset_to_seed(seed)` for the scene and these arrays. `atoms` is what the scene
    classified when it was captured; a restore that classifies differently is refused."""

    model_config = ConfigDict(frozen=True)

    state_id: str
    family: str
    seed: int
    scene_fingerprint: str
    qpos: list[float]
    qvel: list[float]
    ctrl: list[float]
    grip: float
    atoms: list[str]
    cases: list["BenchmarkCase"]
    # How the state was made, for a reader; never used to rebuild it.
    provenance: str = ""


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill: str
    cube: str | None = None
    # SweepShelf only: [offset, speed]. A continuous-parameter skill's cases are its
    # grid over the allowed range (`verdicts.SWEEP_GRID`), one case per setting and
    # target cube.
    params: list[float] = Field(default_factory=list)

    def case_id(self, *, state_id: str) -> str:
        parts = [state_id, self.skill]
        if self.cube is not None:
            parts.append(self.cube)
        if self.params:
            parts.append("p" + "_".join(f"{v:g}" for v in self.params))
        return "-".join(parts)
