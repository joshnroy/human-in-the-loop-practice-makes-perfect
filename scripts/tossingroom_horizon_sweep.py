"""Measure how much of an unpracticed EES's Tossing Room score is retries of the one
stochastic skill rather than competence, as a function of the evaluation horizon.

This exists because the horizon `TossingRoomProblem.max_episode_steps` returns is not a
neutral bookkeeping detail here. `Throw` is the domain's only stochastic skill, and a
failed throw is not terminal -- the robot still holds the item and is still in the bin
room -- so the next policy step simply replans to `Throw` again. Every spare step past
the shortest solve is therefore one more free draw at the ~0.19-probability window a
uniformly random force lands in, and the horizon silently decides how many draws the
evaluation grants. Establishing that costs a measurement, not an argument, which is what
this collects.

**One rollout set yields every horizon, exactly and paired.** `run_task_episode` checks
the goal at the top of each iteration and only then calls the policy, so the number of
policy calls before it returns is exactly the number of actions the episode needed. The
policy replans from the current state with no history, so truncating the horizon to `H`
stops the *same* trajectory earlier: success at `H` is exactly `steps_to_success <= H`.
Running each episode once at the largest horizon under test and reading off prefixes is
therefore not an approximation -- and it avoids the alternative of one run per horizon,
whose RNG streams would diverge after the first extra throw and leave the comparison
unpaired on the very axis being measured.

It lives in `scripts/` because it *drives* simulations, which `analysis/` may never do
(CLAUDE.md); `analysis/practice_makes_perfect/tossingroom_horizon_table.py` renders the
JSON this writes. Seeds are fixed (0..num_seeds-1), never randomly drawn, same as
`scripts/run_sweep.py`.
"""

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from hitl_pmp.core.method.types import LabeledAction, Policy
from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment
from hitl_pmp.environments.tossingroom.problem import TossingRoomProblem
from hitl_pmp.environments.tossingroom.skill_provider import TossingRoomSkillProvider
from hitl_pmp.environments.tossingroom.tasks import TossingRoomTasks
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod


class FixedHorizonProblem(TossingRoomProblem):
    """Pins the horizon, so the probe can drive the real `run_task_episode` at the old
    `2 * num_rooms + 2` value regardless of what the shipped formula returns today. A
    subclass rather than a monkeypatch: the rest of the episode loop must stay exactly
    the one a real evaluation uses, or the measurement is about a copy of the protocol
    instead of the protocol."""

    horizon: int

    def max_episode_steps(self) -> int:
        return self.horizon


class RecordingPolicy(BaseModel):
    """Wraps a task policy and records the skill id executed at each step. A real
    instance, not a static container: it carries the per-episode step log."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner: Policy
    steps: list[int] = Field(default_factory=list)

    # `Policy` is an interface demanding a positional callable, so the single positional
    # argument is unavoidable -- the same exemption CLAUDE.md grants other dunders.
    def __call__(self, state: State) -> LabeledAction:  # noqa: PLR0917
        labeled = self.inner(state)
        # action[0] is the discrete skill id; see TossingRoomEnvironment's docstring.
        self.steps.append(int(round(float(labeled.action[0]))))
        return labeled


class HorizonSweep:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    @staticmethod
    def run_seed(*, seed: int, horizon: int, num_test_tasks: int) -> dict:
        """One unpracticed-EES evaluation sweep at the largest horizon under test.

        Unpracticed means a fresh `EesMethod` that has never had `end_cycle` called --
        the `--num-cycles 0` arm, i.e. the floor the success metric starts from before
        any learning has happened. That floor is what the horizon inflates."""
        env = TossingRoomEnvironment()
        # num_test_tasks must be passed, not left at its default: this domain's test
        # set has a *fixed* goal-family composition, and TossingRoomTasks divides
        # exactly this many tasks up between the families. Drawing more than the field
        # says silently starts a second composition block rather than failing -- 30
        # draws against the default of 10 realises 12/12/6, not the 14/14/2 every other
        # Tossing Room experiment is measured on.
        tasks = TossingRoomTasks(env=env, seed=seed, num_test_tasks=num_test_tasks)
        problem = FixedHorizonProblem(env=env, tasks=tasks, horizon=horizon)
        method = EesMethod(env=env, skill_provider=TossingRoomSkillProvider(env=env), seed=seed)
        problem.hard_reset()
        # Drawn from the test stream in one batch up front, exactly as PracticeLoop does,
        # so the sweep is the protocol's own fixed test set rather than a fresh draw.
        test_tasks: list[Task] = [problem.sample_test_task() for _ in range(num_test_tasks)]

        episodes: list[dict] = []
        for task in test_tasks:
            policy = RecordingPolicy(inner=method.get_task_policy(task=task))
            solved, _ = problem.run_task_episode(task=task, policy=policy)
            episodes.append({
                "solved": bool(solved),
                # Actions taken. When solved, exactly steps-to-success -- which is what
                # makes every shorter horizon derivable from this one rollout.
                "steps": len(policy.steps),
                "skill_ids": list(policy.steps),
            })
        return {"seed": seed, "episodes": episodes}

    @staticmethod
    def collect(*, horizon: int, num_seeds: int, num_test_tasks: int) -> dict:
        return {
            "max_horizon": horizon,
            "num_test_tasks": num_test_tasks,
            "seeds": [
                HorizonSweep.run_seed(seed=seed, horizon=horizon, num_test_tasks=num_test_tasks)
                for seed in range(num_seeds)
            ],
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=16,
        help="Horizon actually rolled out; every smaller one is derived from it.",
    )
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--num-test-tasks", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data = HorizonSweep.collect(
        horizon=args.max_horizon,
        num_seeds=args.num_seeds,
        num_test_tasks=args.num_test_tasks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
