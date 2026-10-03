"""Hand-written protocol fixtures; not VLM, generated-code, or simulator evidence."""

from pathlib import Path
from typing import Any

from hitl_pmp.agentic_runtime.learner import PolicyRevision
from hitl_pmp.agentic_runtime.types import (
    ClusterJudgment,
    ContractJudgment,
    GeneratedLibrary,
    OptionContract,
    RuntimeManifest,
)
from hitl_pmp.agentic_runtime.vision import VLMJudge
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Goal, Task
from hitl_pmp.environments.tossing3d.agentic_bridge import (
    AgenticTossing3DEnvironment,
    Tossing3DAgenticBridge,
)
from hitl_pmp.methods.agentic_options.chain_model import SkillChainModel
from hitl_pmp.methods.agentic_options.method import AgenticOptionsMethod
from hitl_pmp.methods.agentic_options.types import SearchConfig
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    RESET_SKILL,
    TOSS_SKILL,
)
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState


class CountForecast:
    """Exact tiny learning forecast makes the preferred lookahead path checkable."""

    def __init__(self) -> None:
        self.real_observations: list[tuple[str, bool, float]] = []
        self.session_inputs: list[Tossing3DBeliefState] = []

    def success_probability(self, *, belief: Tossing3DBeliefState, skill: str) -> float:
        return 0.5

    def estimated_cost(
        self, *, belief: Tossing3DBeliefState, skill: str, configured_cost: float
    ) -> float:
        return configured_cost

    def observe(
        self,
        *,
        belief: Tossing3DBeliefState,
        skill: str,
        success: bool,
        observed_cost: float | None = None,
    ) -> Tossing3DBeliefState:
        if observed_cost is not None:
            self.real_observations.append((skill, success, observed_cost))
        pending = dict(belief.pending_examples)
        pending[skill] = pending.get(skill, 0) + 1
        return belief.model_copy(update={"pending_examples": pending})

    def advance_session(self, *, belief: Tossing3DBeliefState) -> Tossing3DBeliefState:
        self.session_inputs.append(belief)
        return belief.model_copy(update={"pending_examples": {}})

    def stop_value(self, *, belief: Tossing3DBeliefState, cost: float) -> float:
        return 4.0 * belief.pending_examples.get(TOSS_SKILL, 0) - cost

    def objective(self, *, value: float, cost: float) -> float:
        return value - cost

    def sample_thetas(self, *, belief: Tossing3DBeliefState, count: int) -> Any:
        raise AssertionError("Exact-value search must not sample latent parameters")

    def sample_values(self, *, belief: Tossing3DBeliefState, count: int) -> Any:
        raise AssertionError("Exact-value search must not sample deployment values")

    def evaluate(self, *, theta: Any) -> float:
        raise AssertionError("Exact-value search must not evaluate a sampled parameter")


class RecordingLearner:
    """Tracks real-session callbacks without invoking a coding agent."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.evidence: list[dict[str, Any]] = []

    @property
    def library(self) -> GeneratedLibrary:
        return GeneratedLibrary(
            manifest=library(), controllers={"policies/toss.py": "# unit-test fixture only\n"}
        )

    def begin_session(self) -> None:
        self.events.append("begin")

    def record(self, *, evidence: dict[str, Any]) -> None:
        self.events.append("record")
        self.evidence.append(evidence)

    def end_session(self) -> PolicyRevision:
        self.events.append("learn")
        return PolicyRevision(
            previous_digest="unchanged",
            digest="unchanged",
            changed_files=(),
            evidence_count=len(self.evidence),
        )


class ScriptedJudge(VLMJudge):
    """Deterministic judgments test orchestration; no images are interpreted."""

    def __init__(
        self, *, clusters: list[str | None], judgments: dict[str, bool | None] | None = None
    ) -> None:
        self.clusters = list(clusters)
        self.judgments = judgments or {}
        self.phases: list[str] = []

    def classify(
        self, *, observation: dict[str, Any], manifest: RuntimeManifest
    ) -> ClusterJudgment:
        self.phases.append("classify")
        return ClusterJudgment(cluster_id=self.clusters.pop(0), reason="scripted unit fixture")

    def check(
        self, *, observation: dict[str, Any], option: OptionContract, phase: str
    ) -> ContractJudgment:
        self.phases.append(phase)
        return ContractJudgment(
            value=self.judgments.get(phase, True), reason="scripted unit fixture"
        )


def library() -> RuntimeManifest:
    """Recoverable and unrecoverable bins distinguish physically useful help."""
    options = [
        {
            "option_id": "toss",
            "belief_skill": TOSS_SKILL,
            "controller": "policies/toss.py",
            "cost": 1.0,
        },
        {
            "option_id": "reset_far",
            "belief_skill": RESET_SKILL,
            "human_destination": "far",
            "cost": 5.0,
        },
        {
            "option_id": "reset_near",
            "belief_skill": RESET_SKILL,
            "human_destination": "near",
            "cost": 5.0,
        },
    ]
    edges = [
        {
            "source": "blocked",
            "option_id": f"reset_{side}",
            "outcome": "success",
            "destination": side,
            "probability": 1.0,
        }
        for side in ("far", "near")
    ]
    edges.extend(
        {
            "source": source,
            "option_id": "toss",
            "outcome": outcome,
            "destination": destination,
            "probability": 1.0,
        }
        for source, destination in (("near", "near"), ("far", "blocked"))
        for outcome in ("success", "failure")
    )
    return RuntimeManifest.model_validate({
        "revision": "hand-authored-unit-fixture",
        "clusters": [
            {"cluster_id": name, "description": f"Fixture scene {name}."}
            for name in ("blocked", "near", "far")
        ],
        "options": [
            {
                **option,
                "initiation": "Fixture ready.",
                "termination": "Fixture ended.",
                "success": "Fixture success.",
            }
            for option in options
        ],
        "edges": edges,
    })


def task() -> Task:
    return Task(initial_state=State(data={}), goal=Goal(atoms=frozenset()))


def method(*, tmp_path: Path, judge: ScriptedJudge) -> AgenticOptionsMethod:
    forecast = CountForecast()
    chain = SkillChainModel(manifest=library(), learner_belief=forecast)
    practice_env = AgenticTossing3DEnvironment(scene_bg=False)
    evaluation_env = AgenticTossing3DEnvironment(scene_bg=False)
    return AgenticOptionsMethod(
        env=practice_env,
        evaluation_env=evaluation_env,
        practice_observer=Tossing3DAgenticBridge(env=practice_env),
        evaluation_observer=Tossing3DAgenticBridge(env=evaluation_env),
        evaluation_skill_order=(TOSS_SKILL, PICK_SKILL, OPEN_GRIPPER_SKILL),
        judge=judge,
        learner=RecordingLearner(),
        learner_belief=forecast,
        chain=chain,
        belief=Tossing3DBeliefState(skill_beliefs={}),
        config=SearchConfig(depth=3, observation_probability_weight=0),
        action_ids={"toss": 100},
        human_destinations={"reset_near": "robot_side", "reset_far": "opposite_side"},
        audit_path=tmp_path / "audit.jsonl",
    )
