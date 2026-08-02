"""Post-run analysis for the Tossing Room `Pickup` precondition fix: the actual plan
traces Fast Downward emits, before and after, on the same tasks and seeds.

Reads only already-produced output (CLAUDE.md's analysis/ convention -- never runs a
simulation): a traces JSON of the form
`{"before"|"after": [{"task": i, "solved": bool, "steps": [{"name": str, "bad": bool}]}]}`
where `bad` marks a Pickup scheduled in a room the dynamics refuse to pick up in.

A learning curve is the wrong picture for this defect. Nothing is being *learned*
better -- the planner was emitting plans that could not execute, and the fix makes the
symbolic model exactly as strong as `TossingRoomEnvironment._apply_pickup`'s guard. The
readable evidence is the step sequence itself: before, `Pickup` sits after several
`MoveRoom`s (in the bin room, a silent no-op); after, it is first (in the pile room).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

_COLORS = {
    "Pickup": "tab:blue",
    "MoveRoom": "lightsteelblue",
    "Throw": "tab:green",
    "Press": "tab:purple",
}
_BAD = "tab:red"


class TossingRoomPlanTraces:
    """A static-method container, never instantiated."""

    @staticmethod
    def _draw(*, ax, runs: list[dict], title: str) -> None:
        for row, run in enumerate(runs):
            for col, step in enumerate(run["steps"]):
                bad = step["bad"]
                ax.barh(
                    row,
                    0.92,
                    left=col,
                    height=0.62,
                    color=_BAD if bad else _COLORS.get(step["name"], "grey"),
                    edgecolor="black",
                    linewidth=1.4 if bad else 0.4,
                )
                label = step["name"][:5]
                ax.text(
                    col + 0.46,
                    row,
                    f"{label}✗" if bad else label,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white"
                    if bad or step["name"] in ("Pickup", "Throw", "Press")
                    else "black",
                )
            ax.text(
                len(run["steps"]) + 0.25,
                row,
                "solved" if run["solved"] else "FAILED",
                va="center",
                fontsize=7,
                color="tab:green" if run["solved"] else _BAD,
                fontweight="bold",
            )
        solved = sum(r["solved"] for r in runs)
        ax.set_title(f"{title}  —  {solved}/{len(runs)} solved", fontsize=10)
        ax.set_yticks(range(len(runs)))
        ax.set_yticklabels([f"task {r['task']}" for r in runs], fontsize=7)
        ax.set_xlabel("plan step", fontsize=8)
        ax.set_xlim(-0.2, 7.6)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)

    @staticmethod
    def render(*, traces: Path, output: Path) -> None:
        data = json.loads(traces.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        TossingRoomPlanTraces._draw(
            ax=axes[0],
            runs=data["before"],
            title="BEFORE: Pickup permitted in any room",
        )
        TossingRoomPlanTraces._draw(
            ax=axes[1],
            runs=data["after"],
            title="AFTER: Pickup requires PileInRoom",
        )
        fig.suptitle(
            "Tossing Room: the plans Fast Downward emits, before vs after the "
            "Pickup precondition fix\n"
            "red = a Pickup scheduled where the environment refuses to pick up "
            "(a silent no-op)",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(output, dpi=150)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    TossingRoomPlanTraces.render(traces=args.traces, output=args.output)


if __name__ == "__main__":
    main()
