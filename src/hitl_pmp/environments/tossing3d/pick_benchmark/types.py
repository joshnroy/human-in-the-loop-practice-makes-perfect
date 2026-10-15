"""The Tossing3D pick benchmark's records: a stored state, one run's result, the
per-skill verdict and the report."""

from pydantic import BaseModel, ConfigDict, Field

# How a pick run ended. The pick plans in KINDER's PyBullet scene and executes in
# MuJoCo, with no MuJoCo re-check between (unlike the shelf), so:
#   no_plan          the dispatched controller found no plan (0 control steps)
#   execution_failed it planned, ran in MuJoCo, and the cube is not held -- a PyBullet
#                    plan that failed in MuJoCo is the planning/execution gap
#   budget           it ran to the step limit without terminating
PICK_LABELS = ("success", "no_plan", "execution_failed", "budget", "harness_error")


class PickReport(BaseModel):
    domain: str = "tossing3d"
    pick_implementation: str
    git_commit: str
    git_dirty: bool
    kindergarden_commit: str
    kinder_models_commit: str
    state_set: str
    state_set_sha256: str
    passed: bool
    skills: list["PickVerdict"]
    num_cases: int
    elapsed_seconds: float = 0.0


class PickVerdict(BaseModel):
    skill: str
    # Every run whose state the operator model admits, PickPlannable aside: the robot's
    # hand empty, the bin upright, the cube on the robot's side of the barrier.
    successes: int
    total: int
    passed: bool
    labels: dict[str, int] = Field(default_factory=dict)
    # Runs whose dispatched pick found a plan in KINDER's PyBullet scene.
    pybullet_planned: int = 0
    # The PickPlannable dry run's verdict, and how often it agreed with dispatch.
    dry_run_plannable: int = 0
    dry_run_agrees: int = 0
    # Runs outside the operator's preconditions (tipped bin, ...): run and reported,
    # never counted toward the target.
    outside_operator: int = 0
    per_family: dict[str, list[int]] = Field(default_factory=dict)


class PickResult(BaseModel):
    case_id: str
    state_id: str
    family: str
    skill: str
    label: str
    in_target: bool
    # PickPlannable's dry run of the controller's planning, before dispatch.
    dry_run_plannable: bool | None = None
    dry_run_reason: str | None = None
    pybullet: str = "no_plan"  # "planned" when the dispatched pick took a control step
    controller_steps: int = 0
    terminated: bool = False
    error: str | None = None
    wall_seconds: float = 0.0
    video: str | None = None


class PickState(BaseModel):
    """One start state as KINDER's own object-centric values (`snapshot_to_plain`, at
    MuJoCo's float64 precision): `restore_plain_snapshot` puts a scene back in it with
    no RNG. `atoms` is what it classified as when captured; a restore that classifies
    differently is refused."""

    model_config = ConfigDict(frozen=True)

    state_id: str
    family: str
    seed: int
    layout: str
    scene_fingerprint: str
    plain: dict[str, list[float]]
    atoms: list[str]
    skill: str  # PickCube (cube on the floor) or PickCubeFromBin (cube in the bin)
    in_target: bool
    provenance: str = ""
