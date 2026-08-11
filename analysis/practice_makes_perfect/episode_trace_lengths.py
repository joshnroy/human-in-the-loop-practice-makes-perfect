"""Real per-episode steps-to-goal for the human-ladder rate sweep
(`docs/experiment-logs/2026-08-11-human-ladder-fixed-interval-10x.md`), read from
`--record-episode-traces`' `episode_traces.jsonl` sidecar rather than the narrow,
single-task, final-checkpoint proxy `episode.mp4`'s frame count used to be the only
source for.

**Why this exists.** Josh asked for trace-length analysis (shortest/longest steps-to-goal
per N) on the rate sweep. `core.metrics.types.TaskOutcome` (what `stats.json` carries)
records only solved/unsolved per evaluated task -- never a step count -- so that question
was previously unanswerable from committed data at all. `hitl_pmp.episode_traces`
(this repo's `--record-episode-traces` flag) fixes the underlying data model; this module
is the first real consumer of it.

**One line per (checkpoint, task_index, step) tuple; grouped back into episodes here.**
`load_episode_lengths` walks every `<seed>/episode_traces.jsonl` under an arm directory,
groups rows by `(seed, checkpoint, task_index)`, and reduces each group to one episode:
its family (`GoalFamilies.classify`, reused from `human_ladder_curves.py` rather than
recopied -- same reasoning that module's own docstring gives), whether it solved, and how
many steps it took (`max(step_index) + 1`, contiguity already asserted by
`tests/test_episode_traces.py`). A zero-action episode (an already-satisfied task) writes
no lines at all (`EpisodeTraceRecorder.record_episode`'s own contract) and so is simply
absent, not zero -- consistent with how `episode_traces.py` documents that case.

**Only SOLVED episodes go into the trace-length figures/tables.** An unsolved episode's
step count is `max_episode_steps()` by construction (the loop ran out of budget, not "the
task took this many steps to reach the goal") -- see `EpisodeTrace`'s own docstring on
`core.Problem.run_task_episode`. Pooling unsolved episodes in would silently inflate every
distribution with a constant at the horizon. `solved_rate` is reported alongside the
step-count stats precisely so a reader can see how much of the population was excluded.

**Pooled across every checkpoint and every seed, not just the final one.** The old
`episode.mp4`-frame-count proxy was task-index-0-only, final-checkpoint-only. The point of
carrying real trace data is exactly to stop doing that -- an arm's shortest/longest solve
can appear at any point across the 101 checkpoints x 10 seeds x (14 or 2) tasks per family,
and this pools all of it.

**Optimal-step floors are cross-checked against the data, not assumed.**
`_OPTIMAL_FLOORS` below is not invented for this figure -- it is copied from
`tossingroom_optimal_step_counts.py`'s own already-published, `FastDownwardPlanner`-backed
table (TRASH=5, RECYCLING=4, EMPTY=10 under the one-way layout every arm here except
`two-way-ledge` runs) plus one more entry this module adds: EMPTY=9 under
`--two-way-ledge`, verified directly against `TossingRoomProblem.empty_both_bins_solve()`
for this sweep's exact layout (`--num-rooms 7 --start-room 3 --blocked-right-from 2
--recycling-bin-room 1 --trash-bin-room 6`) -- see this module's own test for the assertion.
`check_floors_against_data` then verifies each floor against the minimum OBSERVED
solved-episode step count in the real trace data and raises rather than silently plotting a
wrong reference line if the two disagree (a real planner bug, a layout mismatch, or a
mis-copied constant would all show up here).
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from analysis.practice_makes_perfect.goal_families import GoalFamilies  # noqa: E402

# Copied from tossingroom_optimal_step_counts.py's own FastDownwardPlanner-backed table
# (the one-way three rows) plus this sweep's own two-way EMPTY floor -- see the module
# docstring for how each was verified, and check_floors_against_data for the runtime
# cross-check against this sweep's own real data.
_OPTIMAL_FLOORS = {
    ("one-way", "TRASH"): 5,
    ("one-way", "RECYCLING"): 4,
    ("one-way", "EMPTY"): 10,
    ("two-way", "TRASH"): 5,
    ("two-way", "RECYCLING"): 4,
    ("two-way", "EMPTY"): 9,
}

_FAMILIES = ("TRASH", "RECYCLING", "EMPTY")

# Fixed report/plot order: the control, the rate sweep low-to-high, then the ceiling.
_ARM_ORDER = ("no-human", "N5", "N7", "N10", "N14", "N20", "N25", "N30", "two-way-ledge")

# Same role colours as human_ladder_curves.py -- reused, not re-derived (CLAUDE.md: "Any
# subagent building a training-curve figure should follow this section without being
# re-briefed" -- the same discipline applies to this figure even though it is a
# distribution plot, not a training curve).
_NO_HUMAN_COLOR = "#D55E00"
_TWO_WAY_COLOR = "#7F7F7F"
_RATE_SWEEP_NS = (5, 7, 10, 14, 20, 25, 30)


class TraceLengths:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    # ------------------------------------------------------------------ reading back

    @staticmethod
    def load_episode_lengths(*, directory: Path) -> list[dict]:
        """Every episode under one arm's `<seed>/episode_traces.jsonl` files, reduced to
        one row each: `{"seed", "checkpoint", "task_index", "family", "solved",
        "num_steps"}`."""
        seed_dirs = sorted(directory.glob("*/episode_traces.jsonl"))
        if not seed_dirs:
            raise ValueError(
                f"no <seed>/episode_traces.jsonl under {directory} -- was this arm run "
                "with --record-episode-traces?"
            )
        grouped: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
        for path in seed_dirs:
            seed = int(path.parent.name)
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (seed, record["checkpoint"], record["task_index"])
                grouped[key].append(record)
        episodes = []
        for (seed, checkpoint, task_index), steps in grouped.items():
            steps.sort(key=lambda step: step["step_index"])
            expected = list(range(len(steps)))
            actual = [step["step_index"] for step in steps]
            if actual != expected:
                raise ValueError(
                    f"{directory} seed={seed} checkpoint={checkpoint} task={task_index}: "
                    f"step_index sequence {actual} is not contiguous from 0 -- a line is "
                    "missing or duplicated."
                )
            solved_values = {step["solved"] for step in steps}
            if len(solved_values) != 1:
                raise ValueError(
                    f"{directory} seed={seed} checkpoint={checkpoint} task={task_index}: "
                    f"solved disagrees across steps ({solved_values})."
                )
            episodes.append({
                "seed": seed,
                "checkpoint": checkpoint,
                "task_index": task_index,
                "family": GoalFamilies.classify(goal=steps[0]["goal"]),
                "solved": steps[0]["solved"],
                "num_steps": len(steps),
            })
        return episodes

    @staticmethod
    def load_all_arms(*, directories: dict[str, Path]) -> dict[str, list[dict]]:
        return {
            name: TraceLengths.load_episode_lengths(directory=d) for name, d in directories.items()
        }

    # ------------------------------------------------------------------ arithmetic

    @staticmethod
    def solved_steps(*, episodes: list[dict], family: str) -> list[int]:
        return [e["num_steps"] for e in episodes if e["family"] == family and e["solved"]]

    @staticmethod
    def solved_rate(*, episodes: list[dict], family: str) -> tuple[int, int]:
        family_episodes = [e for e in episodes if e["family"] == family]
        return sum(1 for e in family_episodes if e["solved"]), len(family_episodes)

    @staticmethod
    def layout(*, arm: str) -> str:
        return "two-way" if arm == "two-way-ledge" else "one-way"

    @staticmethod
    def check_floors_against_data(*, arms: dict[str, list[dict]]) -> list[str]:
        """Returns a list of human-readable disagreements (empty if every floor matches
        the minimum OBSERVED solved-episode step count) -- checked, not assumed, per this
        module's own docstring. A floor may legitimately be strictly below the observed
        minimum (no arm here is guaranteed to have found the truly optimal solve at every
        checkpoint), but the observed minimum must never be BELOW the floor -- that would
        mean either the floor or the domain layout used to compute it is wrong."""
        problems = []
        for arm, episodes in arms.items():
            layout = TraceLengths.layout(arm=arm)
            for family in _FAMILIES:
                steps = TraceLengths.solved_steps(episodes=episodes, family=family)
                if not steps:
                    continue
                floor = _OPTIMAL_FLOORS[(layout, family)]
                observed_min = min(steps)
                if observed_min < floor:
                    problems.append(
                        f"{arm}/{family}: observed min solved-episode length "
                        f"{observed_min} is BELOW the published floor {floor} "
                        f"({layout} layout) -- the floor or the layout is wrong."
                    )
        return problems

    # ------------------------------------------------------------------ reporting

    @staticmethod
    def print_report(*, arms: dict[str, list[dict]]) -> None:
        problems = TraceLengths.check_floors_against_data(arms=arms)
        if problems:
            print("FLOOR CROSS-CHECK FAILURES:")
            for problem in problems:
                print(f"  ! {problem}")
        else:
            print("Floor cross-check: every published floor is <= the observed minimum.")
        print()
        for arm in _ARM_ORDER:
            if arm not in arms:
                continue
            episodes = arms[arm]
            layout = TraceLengths.layout(arm=arm)
            print(f"{arm} ({layout}):")
            for family in _FAMILIES:
                steps = TraceLengths.solved_steps(episodes=episodes, family=family)
                solved, total = TraceLengths.solved_rate(episodes=episodes, family=family)
                floor = _OPTIMAL_FLOORS[(layout, family)]
                if not steps:
                    print(f"  {family:>10}  {solved}/{total} solved -- no solved episodes")
                    continue
                print(
                    f"  {family:>10}  {solved}/{total} solved  "
                    f"min={min(steps)} (floor {floor})  "
                    f"median={statistics.median(steps):.1f}  max={max(steps)}"
                )
            print()

    # ------------------------------------------------------------------ figure

    @staticmethod
    def arm_color(*, arm: str) -> str:
        if arm == "no-human":
            return _NO_HUMAN_COLOR
        if arm == "two-way-ledge":
            return _TWO_WAY_COLOR
        n = int(arm[1:])
        cmap = matplotlib.colormaps["Blues"]
        norm = matplotlib.colors.Normalize(vmin=min(_RATE_SWEEP_NS), vmax=max(_RATE_SWEEP_NS))
        return cmap(0.30 + 0.65 * norm(n))

    @staticmethod
    def render(*, arms: dict[str, list[dict]], output: Path, title: str) -> None:
        """Three panels (TRASH/RECYCLING/EMPTY), one figure. Each panel: one box per arm
        (fixed `_ARM_ORDER`) of solved-episode step counts, pooled across every checkpoint
        and every seed -- the real shortest/longest steps-to-goal, not the single-task
        final-checkpoint proxy. Faint jittered points underneath each box show the actual
        spread rather than only the box's five-number summary. A dashed reference line
        marks the optimal floor for that family/layout -- two lines on the EMPTY panel,
        since `two-way-ledge` genuinely solves it in fewer steps (a real domain-difficulty
        difference, not a method effect -- see `TossingRoomProblem.empty_both_bins_solve`'s
        own docstring)."""
        present_arms = [arm for arm in _ARM_ORDER if arm in arms]
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), sharey=False)
        rng_seed = 0
        for ax, family in zip(axes, _FAMILIES, strict=True):
            box_data = []
            colors = []
            denominators = []
            for arm in present_arms:
                steps = TraceLengths.solved_steps(episodes=arms[arm], family=family)
                solved, total = TraceLengths.solved_rate(episodes=arms[arm], family=family)
                box_data.append(steps if steps else [float("nan")])
                colors.append(TraceLengths.arm_color(arm=arm))
                denominators.append((solved, total))

            positions = list(range(1, len(present_arms) + 1))
            bp = ax.boxplot(
                box_data,
                positions=positions,
                widths=0.55,
                showfliers=False,
                patch_artist=True,
                medianprops={"color": "black", "linewidth": 1.6},
                whiskerprops={"linewidth": 1.1},
                capprops={"linewidth": 1.1},
                boxprops={"linewidth": 1.1},
            )
            for patch, color in zip(bp["boxes"], colors, strict=True):
                patch.set_facecolor(color)
                patch.set_alpha(0.55)

            # Deterministic jitter (seeded, not drawn per-run) so the figure is
            # byte-reproducible from the same underlying data.
            import random

            jitter_rng = random.Random(rng_seed)
            for position, steps, color in zip(positions, box_data, colors, strict=True):
                if len(steps) == 1 and steps[0] != steps[0]:  # nan sentinel, no data
                    continue
                xs = [position + jitter_rng.uniform(-0.18, 0.18) for _ in steps]
                ax.scatter(xs, steps, s=8, color=color, alpha=0.16, zorder=1, linewidths=0)

            layouts_present = {TraceLengths.layout(arm=arm) for arm in present_arms}
            for layout in sorted(layouts_present):
                floor = _OPTIMAL_FLOORS[(layout, family)]
                arm_positions = [
                    positions[i]
                    for i, arm in enumerate(present_arms)
                    if TraceLengths.layout(arm=arm) == layout
                ]
                ax.hlines(
                    floor,
                    min(arm_positions) - 0.4,
                    max(arm_positions) + 0.4,
                    color="black",
                    linestyle=(0, (4, 2)),
                    linewidth=1.1,
                    zorder=0,
                    label=f"optimal floor, {layout} ({floor})",
                )

            ax.set_xticks(positions)
            ax.set_xticklabels(present_arms, rotation=45, ha="right", fontsize=8)
            total_solved = sum(solved for solved, _ in denominators)
            total_possible = sum(total for _, total in denominators)
            ax.set_title(
                f"{family} (solved episodes: {total_solved}/{total_possible} pooled across arms)",
                fontsize=9.5,
            )
            ax.set_ylabel("steps to goal (solved episodes only)", fontsize=8.5)
            ax.grid(alpha=0.25, linewidth=0.6, axis="y")
            ax.legend(fontsize=7, loc="upper right", framealpha=0.95)

        fig.suptitle(title, fontsize=10.5)
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        print(f"wrote {output}")
        plt.close(fig)

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--arm",
            action="append",
            required=True,
            metavar="NAME=DIR",
            help="e.g. no-human=.../no-human/ees, N7=.../rate-sweep/N7/ees, "
            "two-way-ledge=.../two-way-ledge/ees . DIR holds <seed>/episode_traces.jsonl "
            "(a --record-episode-traces run).",
        )
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument(
            "--title",
            default="Tossing Room human ladder -- real steps-to-goal per N "
            "(pooled across every checkpoint and seed)",
        )
        args = parser.parse_args()

        directories = {}
        for spec in args.arm:
            name, _, path = spec.partition("=")
            directories[name] = Path(path)

        arms = TraceLengths.load_all_arms(directories=directories)
        TraceLengths.print_report(arms=arms)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        TraceLengths.render(arms=arms, output=args.output, title=args.title)


if __name__ == "__main__":
    TraceLengths.main()
