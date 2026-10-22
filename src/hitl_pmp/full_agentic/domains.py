"""What the Full-Agentic runner needs from a domain, and the domains it can run.

The runner, the persistent `World` and the frozen evaluation are domain-free; each
`--env` supplies its environment, its agentic bridge (the LLM-facing robot API), its
prompt files, its human interventions and its goal check through one of these. The
registry follows `cli.py`'s `STEP_PROTOCOL_ENVIRONMENTS`.
"""

import abc
import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from hitl_pmp.core.practice_costs import HumanCharge, PracticeCosts
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.sweepoffshelf3d.agentic_bridge import SweepOffShelf3DAgenticBridge
from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.skills import RESET_CUBES_NAME, RESET_WIPER_NAME
from hitl_pmp.environments.sweepoffshelf3d.tasks import SweepOffShelf3DTasks
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
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        """The domain's own runner flags, if any."""
        del parser

    @staticmethod
    def environment_options(*, args: argparse.Namespace) -> dict[str, Any]:
        """JSON-serializable keyword arguments for `make_environment`."""
        return {}

    @staticmethod
    @abc.abstractmethod
    def make_environment(*, options: dict[str, Any], human_seed: int) -> Any:
        """Practice uses `human_seed=seed`, evaluation `seed + 1`, as `--env`'s other
        methods do."""

    @staticmethod
    @abc.abstractmethod
    def make_tasks(*, env: Any, seed: int) -> Any: ...

    @staticmethod
    @abc.abstractmethod
    def make_bridge(*, env: Any, step_limit: int) -> Any:
        """An object-state bridge that refuses commands past `step_limit`: practice passes
        `PRACTICE_STEP_LIMIT` (its world counts the budget), an evaluation its own budget,
        so the bridge never ends a held-out task before the scoring clock does."""

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
    def protocol_fields(*, options: dict[str, Any]) -> dict[str, Any]:
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
    def make_environment(*, options: dict[str, Any], human_seed: int) -> Any:
        # Tossing3D's human resets draw from the environment's own seed, as before.
        del options, human_seed
        return Tossing3DEnvironment()

    @staticmethod
    def make_tasks(*, env: Any, seed: int) -> Any:
        return Tossing3DTasks(env=env, seed=seed)

    @staticmethod
    def make_bridge(*, env: Any, step_limit: int) -> Any:
        return Tossing3DAgenticBridge(
            env=env, observation_mode="object_state", step_limit=step_limit
        )

    @staticmethod
    def goal_check(*, env: Any) -> Callable[[], bool]:
        return env.backend().check_goals

    @staticmethod
    def task_seed(*, env: Any, task: Task) -> int:
        return int(task.initial_state.get(obj=env.scene, feature_name="seed"))

    @staticmethod
    def default_costs() -> PracticeCosts:
        return PracticeCosts()


class SweepOffShelf3DFullAgentic(FullAgenticDomain):
    name = "sweepoffshelf3d"
    task_prompt_file = "sweepoffshelf3d_task_prompt.md"
    robot_api_file = "sweepoffshelf3d_robot_api.md"
    human_interventions = {
        RESET_CUBES_NAME: RESET_CUBES_NAME,
        RESET_WIPER_NAME: RESET_WIPER_NAME,
    }

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--reset-cubes-success-probability",
            type=float,
            default=SweepOffShelf3DEnvironment.model_fields[
                "reset_cubes_success_probability"
            ].default,
            help="p: the human's true competence at reset_cubes.",
        )

    @staticmethod
    def environment_options(*, args: argparse.Namespace) -> dict[str, Any]:
        return {"reset_cubes_success_probability": args.reset_cubes_success_probability}

    @staticmethod
    def make_environment(*, options: dict[str, Any], human_seed: int) -> Any:
        return SweepOffShelf3DEnvironment(**options, human_seed=human_seed)

    @staticmethod
    def make_tasks(*, env: Any, seed: int) -> Any:
        return SweepOffShelf3DTasks(env=env, seed=seed)

    @staticmethod
    def make_bridge(*, env: Any, step_limit: int) -> Any:
        return SweepOffShelf3DAgenticBridge(env=env, step_limit=step_limit)

    @staticmethod
    def goal_check(*, env: Any) -> Callable[[], bool]:
        # KINDER's own verdict, the one `SweepOffShelf3DStepProtocol.goal_check` scores
        # every planner arm on: each cube wholly inside the upright box, held or not.
        return lambda: bool(env.backend().core()._check_goals())  # noqa: SLF001

    @staticmethod
    def task_seed(*, env: Any, task: Task) -> int:
        return int(task.initial_state.get(obj=env.scene, feature_name="seed"))

    @staticmethod
    def default_costs() -> PracticeCosts:
        return PracticeCosts(
            human_skills={RESET_CUBES_NAME: HumanCharge(), RESET_WIPER_NAME: HumanCharge()}
        )

    @staticmethod
    def protocol_fields(*, options: dict[str, Any]) -> dict[str, Any]:
        return {"env": SweepOffShelf3DFullAgentic.name, "environment": options}


FULL_AGENTIC_DOMAINS: dict[str, type[FullAgenticDomain]] = {
    Tossing3DFullAgentic.name: Tossing3DFullAgentic,
    SweepOffShelf3DFullAgentic.name: SweepOffShelf3DFullAgentic,
}
