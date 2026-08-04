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

**Run this once per horizon.** An earlier version of this script rolled out once at the
largest horizon and derived every shorter one by truncating each recorded trajectory to
its first `H` actions, on the argument that the policy replans from the current state
with no history, so a shorter horizon stops the *same* trajectory earlier and success at
`H` is exactly `steps_to_success <= H`.

**That argument is wrong, and was measured to be wrong.** It holds only for the *first*
episode of each seed. `EesMethod` draws its skill parameters from a single per-run RNG
stream shared across the whole sweep, and a longer horizon issues more `Throw` actions
-- so by episode 2 the longer rollout has consumed a different number of draws and every
subsequent episode sees different sampled forces. Rolled out directly at `H = 7` this
script reproduces the ten-seed EES arms' own first evaluation sweep **seed for seed, all
ten**; derived by truncation from `H = 16` it disagrees on six of the ten. The truncated
estimate is still unbiased for the population quantity (the extra draws are just other
draws from the same distribution), but it is not the run a real evaluation at that
horizon would produce, and it is not paired across horizons in any useful sense either.

So `--max-horizon` is now simply *the* horizon: run the script once per horizon and
compare the resulting files. Separate rollouts are unpaired across horizons, which is a
real cost, but an unpaired comparison of correct numbers beats a paired comparison of
numbers no run produces.

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
        help=(
            "The horizon to roll out at. Run once per horizon and compare the files -- "
            "shorter horizons must NOT be derived by truncating a longer rollout; see "
            "this module's docstring for the measurement that rules that out."
        ),
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
