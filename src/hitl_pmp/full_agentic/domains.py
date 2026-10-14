"""What the Full-Agentic runner needs from a domain, and the domains it can run.

The runner, the persistent `World` and the frozen evaluation are domain-free; each
`--env` supplies its environment, its agentic bridge (the LLM-facing robot API), its
prompt files, its human interventions and its goal check through one of these. The
registry follows `cli.py`'s `STEP_PROTOCOL_ENVIRONMENTS`.
"""

import abc
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from hitl_pmp.core.practice_costs import PracticeCosts
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.tossing3d.agentic_bridge import Tossing3DAgenticBridge
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.tasks import Tossing3DTasks

HERE = Path(__file__).resolve().parent
# A practice bridge is one unbounded session; the World's step budget is the bound.
PRACTICE_STEP_LIMIT = 2**63 - 1


class FullAgenticDomain(abc.ABC):
    """A static-method container: one subclass per `--env`, never instantiated."""

    name: ClassVar[str]
    # Prompt files beside this module, read verbatim.
    task_prompt_file: ClassVar[str]
    robot_api_file: ClassVar[str]
    # intervention_id -> the `reset_movables` destination it dispatches. Iteration
    # order is the order of the World's per-destination counters.
    human_interventions: ClassVar[dict[str, str]]

    @staticmethod
    @abc.abstractmethod
    def make_environment() -> Any: ...

    @staticmethod
    @abc.abstractmethod
    def make_tasks(*, env: Any, seed: int) -> Any: ...

    @staticmethod
    @abc.abstractmethod
    def make_bridge(*, env: Any, practice: bool) -> Any:
        """An object-state bridge: unbounded for practice, default-bounded for one
        evaluation task (whose own clock stops it first)."""

    @staticmethod
    @abc.abstractmethod
    def goal_check(*, env: Any) -> Callable[[], bool]: ...

    @staticmethod
    @abc.abstractmethod
    def task_seed(*, env: Any, task: Task) -> int: ...

    @staticmethod
    @abc.abstractmethod
    def default_costs() -> PracticeCosts: ...

    @staticmethod
    def protocol_fields() -> dict[str, Any]:
        """Extra `protocol.json` fields, appended after the shared ones."""
        return {}

    @classmethod
    def task_prompt(cls) -> str:
        return (HERE / cls.task_prompt_file).read_text()

    @classmethod
    def robot_api(cls) -> Path:
        return HERE / cls.robot_api_file


class Tossing3DFullAgentic(FullAgenticDomain):
    name = "tossing3d"
    task_prompt_file = "task_prompt.md"
    robot_api_file = "robot_api.md"
    human_interventions = {
        "reset_cube_and_bin_near": "robot_side",
        "reset_cube_far": "opposite_side",
    }

    @staticmethod
    def make_environment() -> Any:
        return Tossing3DEnvironment()

    @staticmethod
    def make_tasks(*, env: Any, seed: int) -> Any:
        return Tossing3DTasks(env=env, seed=seed)

    @staticmethod
    def make_bridge(*, env: Any, practice: bool) -> Any:
        if practice:
            return Tossing3DAgenticBridge(
                env=env, observation_mode="object_state", step_limit=PRACTICE_STEP_LIMIT
            )
        return Tossing3DAgenticBridge(env=env, observation_mode="object_state")

    @staticmethod
    def goal_check(*, env: Any) -> Callable[[], bool]:
        return env.backend().check_goals

    @staticmethod
    def task_seed(*, env: Any, task: Task) -> int:
        return int(task.initial_state.get(obj=env.scene, feature_name="seed"))

    @staticmethod
    def default_costs() -> PracticeCosts:
        return PracticeCosts()


FULL_AGENTIC_DOMAINS: dict[str, type[FullAgenticDomain]] = {
    Tossing3DFullAgentic.name: Tossing3DFullAgentic,
}
