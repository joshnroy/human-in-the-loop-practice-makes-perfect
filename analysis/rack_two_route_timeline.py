"""Practice timelines of two-route Serving runs: one row per run, one bar per skill.

Reads each run's `step_events.jsonl` (post-run only). Bars are robot skills over
practice control steps, coloured by skill; bulk skills carry the plate count they left;
the person's batch returns and tool recoveries are vertical lines; a planner STOP is a
triangle at the step it happened; a skill the practice budget cut off is hatched.

    python -m analysis.rack_two_route_timeline --run "EES=<dir>" --run "DP=<dir>" \
        --title "..." --output timeline.png
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SKILL_COLOURS = {
    "PickScraper": "#7f7f7f",
    "ReturnScraper": "#c7c7c7",
    "Sweep": "#E69F00",
    "GatherAndSweep": "#4c72b0",
    "PickFood": "#8c564b",
    "ReturnFoodToRack": "#c49c94",
}
HUMAN_COLOURS = {"human_batch_return": "#000000", "human_tool_recovery": "#c44e52"}
HUMAN_LABELS = {
    "human_batch_return": "human returns the dumplings",
    "human_tool_recovery": "human puts the scraper back",
}
STOP_COLOUR = "#8172b2"


class RackTwoRouteTimeline:
    @staticmethod
    def events(*, run: Path) -> list[dict[str, Any]]:
        path = run / "step_events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    @staticmethod
    def draw(*, runs: list[tuple[str, Path]], title: str, output: Path) -> None:
        loaded = [(label, RackTwoRouteTimeline.events(run=run)) for label, run in runs]
        horizon = max(
            (
                e.get("end_step") or e.get("practice_steps") or 0
                for _, events in loaded
                for e in events
            ),
            default=1,
        )
        figure, axes = plt.subplots(
            len(loaded), 1, figsize=(18, 2.3 * len(loaded) + 1.6), sharex=True, squeeze=False
        )
        seen_skills: list[str] = []
        seen_humans: list[str] = []
        stopped = False
        for axis, (label, events) in zip(axes[:, 0], loaded, strict=True):
            for event in events:
                kind = event.get("event")
                if kind == "robot":
                    skill = event["label"].split("(")[0]
                    if skill not in seen_skills:
                        seen_skills.append(skill)
                    start, end = event["start_step"], event["end_step"]
                    axis.barh(
                        0,
                        end - start,
                        left=start,
                        height=0.6,
                        color=SKILL_COLOURS.get(skill, "#bbbbbb"),
                        edgecolor="white",
                        linewidth=0.8,
                        hatch="//" if event.get("interrupted") else None,
                    )
                    count = (event.get("skill_trace") or {}).get("plate_count")
                    if skill in ("Sweep", "GatherAndSweep") and end - start > horizon / 60:
                        text = "cut off" if event.get("interrupted") else f"{count}"
                        axis.text(
                            (start + end) / 2,
                            0,
                            text,
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white",
                        )
                elif kind == "human":
                    name = event["destination"]
                    if name not in seen_humans:
                        seen_humans.append(name)
                    axis.vlines(
                        event["start_step"],
                        -0.48,
                        0.48,
                        color=HUMAN_COLOURS.get(name, "#000000"),
                        linewidth=2,
                    )
                elif kind == "stop":
                    stopped = True
                    axis.plot(
                        event["start_step"],
                        0.55,
                        marker="v",
                        color=STOP_COLOUR,
                        markersize=10,
                        clip_on=False,
                    )
            axis.set_title(label, loc="left", fontsize=11)
            axis.set_ylim(-0.6, 0.75)
            axis.set_yticks([])
            for side in ("left", "right", "top"):
                axis.spines[side].set_visible(False)
        axes[-1, 0].set_xlim(0, horizon)
        axes[-1, 0].set_xlabel("Practice control steps")
        handles: list[Any] = [
            Patch(color=SKILL_COLOURS.get(s, "#bbbbbb"), label=s) for s in seen_skills
        ]
        handles += [
            Line2D([0], [0], color=HUMAN_COLOURS[h], linewidth=2, label=HUMAN_LABELS[h])
            for h in seen_humans
        ]
        if stopped:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="v",
                    color=STOP_COLOUR,
                    linestyle="",
                    markersize=9,
                    label="planner chose to stop",
                )
            )
        handles.append(
            Patch(facecolor="#dddddd", hatch="//", label="cut off by the practice budget")
        )
        figure.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 5),
            frameon=False,
            fontsize=9,
        )
        figure.suptitle(
            title + "\nnumbers on the bulk skills: dumplings on the plate when the skill returned",
            fontsize=12,
        )
        figure.tight_layout(rect=(0, 0.09, 1, 0.94))
        figure.savefig(output, dpi=110)
        plt.close(figure)

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIRECTORY")
        parser.add_argument("--title", required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        runs = []
        for item in args.run:
            label, _, directory = item.partition("=")
            runs.append((label, Path(directory)))
        RackTwoRouteTimeline.draw(runs=runs, title=args.title, output=args.output)


if __name__ == "__main__":
    RackTwoRouteTimeline.main()
