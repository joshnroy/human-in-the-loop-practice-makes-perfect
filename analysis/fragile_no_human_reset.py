"""Figures for the fragile Tossing3D no-human-reset log: actual endpoints, never filled.

Two modes. ``--artifacts-root`` reads every run directory listed in ``ARMS`` and writes
``curves.json`` into ``--log-dir``; without it the committed ``curves.json`` is re-plotted.
"""

import argparse
import datetime
import json
from pathlib import Path
from typing import Any


class FragileNoHumanReset:
    # Colour carries the method and nothing else (CLAUDE.md, "Training-curve style").
    # The POMDP Planner's two solvers are one method with a knob, so A* is a darker shade
    # of Model B's blue. The four agentic learners keep the colours of the existing
    # fragile comparison figures.
    COLORS = {
        "ees": "#009E73",
        "expectimax": "#0072B2",
        "astar": "#004C7A",
        "hybrid": "#D55E00",
        "llcc": "#6A3D9A",
        "original": "#CC79A7",
        "new-wording": "#E69F00",
    }
    LABELS = {
        "ees": "EES",
        "expectimax": "POMDP Planner / expectimax",
        "astar": "POMDP Planner / A*",
        "hybrid": "POMDP + Agentic Skills",
        "llcc": "POMDP + Agentic Skills & States",
        "original": "Full-Agentic / original",
        "new-wording": "Full-Agentic / new wording",
    }
    NO_HUMAN_LINESTYLE = (0, (4, 2))
    PANELS = (
        ("Structured planners", ("ees", "expectimax", "astar")),
        ("Agentic learners", ("hybrid", "llcc", "original", "new-wording")),
    )
    # (key, human reset offered during practice, seed, loader kind, path under artifacts/)
    ARMS: tuple[tuple[str, bool, int, str, str], ...] = (
        ("ees", False, 0, "structured", "fragile-nohuman-20261013/della/ees/results/ees/0"),
        (
            "expectimax",
            False,
            0,
            "structured",
            "fragile-nohuman-20261013/della/expectimax/results/pomdp/0",
        ),
        ("astar", False, 0, "structured", "fragile-nohuman-20261013/della/astar/results/pomdp/0"),
        (
            "hybrid",
            False,
            0,
            "structured",
            "fragile-nohuman-lccc-20261013/results/pomdp-agentic-skills/0",
        ),
        (
            "llcc",
            False,
            0,
            "structured",
            "fragile-nohuman-llcc-20261013/results/pomdp-agentic-skills-states/0",
        ),
        ("original", False, 0, "full_agentic", "fragile-nohuman-fullagentic-20261013/original/run"),
        (
            "new-wording",
            False,
            0,
            "full_agentic",
            "fragile-nohuman-fullagentic-20261013/new-wording/run",
        ),
        ("ees", True, 0, "structured", "fragile-mat-comparison-notion-20261009/report/ees"),
        (
            "expectimax",
            True,
            0,
            "structured",
            "fragile-mat-comparison-notion-20261009/report/expectimax",
        ),
        *(
            (
                "astar",
                True,
                s,
                "structured",
                f"fragile-comparison-final-20261011/della/seed-{s}/results/pomdp/{s}",
            )
            for s in range(5)
        ),
        (
            "hybrid",
            True,
            0,
            "structured",
            "hybrid-expectimax-agentic-seed0-20261009-r3/results/pomdp-agentic-skills/0",
        ),
        (
            "llcc",
            True,
            0,
            "structured",
            "llcc-fragile-seed0-20261010-r2/results/pomdp-agentic-skills-states/0",
        ),
        (
            "original",
            True,
            0,
            "full_agentic",
            "fragile-tossing-mat1-heavy-20261008/agentic-original-h100-d10/run",
        ),
        (
            "new-wording",
            True,
            0,
            "full_agentic",
            "fragile-tossing-mat1-heavy-20261008/agentic-new-wording-h100-d10/run",
        ),
    )

    @staticmethod
    def _completed_evaluations(*, root: Path, steps_by_index: dict[int, int]) -> list[list[int]]:
        evaluations = []
        for index in sorted(steps_by_index):
            results = root / "evaluations" / f"{index:04d}" / "results.json"
            if not results.exists():
                continue
            record = json.loads(results.read_text())
            if record.get("complete") is False:
                continue
            evaluations.append([steps_by_index[index], record["num_solved"], record["num_total"]])
        return evaluations

    @staticmethod
    def _close_damage(*, events: list[list[int]], endpoint: int, expected: int) -> list[list[int]]:
        if events[-1][1] != expected:
            raise ValueError(f"damage events sum to {events[-1][1]}, status.json says {expected}")
        if events[-1][0] > endpoint:
            raise ValueError(
                f"damage event at step {events[-1][0]} is past the endpoint {endpoint}"
            )
        if events[-1][0] != endpoint:
            events.append([endpoint, events[-1][1]])
        return events

    @staticmethod
    def load_structured_run(*, root: Path) -> dict[str, Any]:
        status = json.loads((root / "status.json").read_text())
        endpoint = int(status["practice_steps"])
        completed = root / "step_evaluations.json"
        if completed.exists():
            evaluations = [
                [r["practice_steps"], r["num_solved"], r["num_total"]]
                for r in json.loads(completed.read_text())
                if r.get("complete", True)
            ]
        else:
            measurements = json.loads((root / "measurements.json").read_text())
            evaluations = FragileNoHumanReset._completed_evaluations(
                root=root, steps_by_index={m["index"]: m["practice_steps"] for m in measurements}
            )
        damage = [[0, 0]]
        robot_invocations = 0
        endpoint_reason = None
        with (root / "step_events.jsonl").open() as stream:
            for line in stream:
                event = json.loads(line)
                if event["event"] == "damage":
                    damage.append([event["practice_step"], damage[-1][1] + len(event["impacts"])])
                elif event["event"] == "robot":
                    robot_invocations += 1
                elif event["event"] == "practice_end":
                    endpoint_reason = event["reason"]
        return dict(
            evaluations=evaluations,
            damage_events=FragileNoHumanReset._close_damage(
                events=damage, endpoint=endpoint, expected=int(status["damage_contacts"])
            ),
            endpoint_steps=endpoint,
            endpoint_reason=endpoint_reason,
            complete=status["phase"] == "complete",
            robot_invocations=robot_invocations,
            human_invocations=int(status.get("human_invocations", 0)),
        )

    @staticmethod
    def load_full_agentic_run(*, root: Path) -> dict[str, Any]:
        status = json.loads((root / "status.json").read_text())
        endpoint = int(status["practice_steps"])
        steps_by_index = {}
        for snapshot in sorted((root / "measurements").glob("*/snapshot.json")):
            record = json.loads(snapshot.read_text())
            steps_by_index[int(record["index"])] = int(record["practice_steps"])
        damage = [[0, 0]]
        with (root / "practice.jsonl").open() as stream:
            for line in stream:
                result = json.loads(line).get("response", {}).get("result", {})
                if not isinstance(result, dict) or "damage_contacts" not in result:
                    continue
                count = int(result["damage_contacts"])
                if count < damage[-1][1]:
                    raise ValueError("damage count decreased within practice.jsonl")
                if count > damage[-1][1]:
                    damage.append([int(result["practice_steps"]), count])
        return dict(
            evaluations=FragileNoHumanReset._completed_evaluations(
                root=root, steps_by_index=steps_by_index
            ),
            damage_events=FragileNoHumanReset._close_damage(
                events=damage, endpoint=endpoint, expected=int(status["damage_contacts"])
            ),
            endpoint_steps=endpoint,
            endpoint_reason=status.get("endpoint"),
            complete=status["phase"] == "complete",
            robot_invocations=None,
            human_invocations=int(status.get("human_requests", 0)),
        )

    @staticmethod
    def extract(*, artifacts_root: Path) -> dict[str, Any]:
        loaders = {
            "structured": FragileNoHumanReset.load_structured_run,
            "full_agentic": FragileNoHumanReset.load_full_agentic_run,
        }
        arms = []
        for key, human_reset, seed, kind, relative in FragileNoHumanReset.ARMS:
            run = loaders[kind](root=artifacts_root / relative)
            arms.append(
                dict(
                    key=key,
                    method=FragileNoHumanReset.LABELS[key],
                    human_reset=human_reset,
                    seed=seed,
                    **run,
                )
            )
        snapshot = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
        return dict(snapshot=snapshot, arms=arms)

    @staticmethod
    def _series(*, arm: dict[str, Any], metric: str) -> tuple[list[int], list[int]]:
        rows = arm["evaluations"] if metric == "performance" else arm["damage_events"]
        return [r[0] for r in rows], [r[1] for r in rows]

    @staticmethod
    def _draw_group(
        *, ax: Any, arms: list[dict[str, Any]], metric: str
    ) -> list[tuple[dict[str, Any], float, float]]:
        import numpy as np

        key = arms[0]["key"]
        human = arms[0]["human_reset"]
        colour = FragileNoHumanReset.COLORS[key]
        style = "-" if human else FragileNoHumanReset.NO_HUMAN_LINESTYLE
        step = metric != "performance"
        draw = ax.step if step else ax.plot
        kwargs = {"where": "post"} if step else {}
        if len(arms) > 1:
            for arm in arms:
                x, y = FragileNoHumanReset._series(arm=arm, metric=metric)
                draw(x, y, color=colour, linestyle=style, linewidth=0.8, alpha=0.16, **kwargs)
            grid = np.unique(
                np.concatenate([FragileNoHumanReset._series(arm=a, metric=metric)[0] for a in arms])
            )
            stacked = []
            for arm in arms:
                x, y = (np.array(v) for v in FragileNoHumanReset._series(arm=arm, metric=metric))
                stacked.append(y[np.searchsorted(x, grid, side="right") - 1])
            x_mean, y_mean = grid, np.mean(stacked, axis=0)
            label = f"{arms[0]['method']} — human reset, mean, n={len(arms)}"
        else:
            x_mean, y_mean = (
                np.array(v) for v in FragileNoHumanReset._series(arm=arms[0], metric=metric)
            )
            status = "" if arms[0]["complete"] else " (in progress)"
            label = (
                f"{arms[0]['method']} — {'human reset' if human else 'no human reset'}, n=1{status}"
            )
        draw(x_mean, y_mean, color=colour, linestyle=style, linewidth=2.3, label=label, **kwargs)
        complete = all(a["complete"] for a in arms)
        marker = (
            dict(marker="D", markersize=6)
            if complete
            else dict(marker="o", markersize=7, markerfacecolor="white")
        )
        ax.plot([x_mean[-1]], [y_mean[-1]], color=colour, linestyle="none", **marker)
        return [(arms[0], float(x_mean[-1]), float(y_mean[-1]))]

    @staticmethod
    def plot(*, curves: dict[str, Any], output_dir: Path, stem: str) -> list[Path]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
        output_dir.mkdir(parents=True, exist_ok=True)
        written = []
        metrics = (
            (
                "performance",
                "solved per seed",
                "held-out tasks solved (of 10) vs. counted practice steps",
            ),
            (
                "raw-ground-hits",
                "cumulative raw-ground hits per seed",
                "cumulative raw-ground hits vs. counted practice steps",
            ),
        )
        present = {a["key"] for a in curves["arms"]}
        panels = [(t, [k for k in keys if k in present]) for t, keys in FragileNoHumanReset.PANELS]
        panels = [(t, keys) for t, keys in panels if keys]
        for metric, ylabel, title in metrics:
            fig, axes = plt.subplots(
                1, len(panels), figsize=(7.2 * len(panels), 6.6), squeeze=False
            )
            for ax, (panel_title, keys) in zip(axes[0], panels, strict=True):
                endpoints = []
                for key in keys:
                    for human in (True, False):
                        arms = [
                            a
                            for a in curves["arms"]
                            if a["key"] == key and a["human_reset"] == human
                        ]
                        if arms:
                            endpoints.extend(
                                FragileNoHumanReset._draw_group(ax=ax, arms=arms, metric=metric)
                            )
                x_max = max(x for _, x, _ in endpoints)
                ax.set_xlim(0, x_max * 1.08)
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
                ax.set_xlabel("counted practice steps")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{panel_title} — Tossing3D fragile, 1 m mat", loc="left", fontsize=11)
                ax.grid(alpha=0.2)
                ax.set_axisbelow(True)
                if metric == "performance":
                    ax.set_ylim(-0.4, 10.6)
                    ax.set_yticks(range(0, 11, 2))
                shown = sorted(
                    (e for e in endpoints if not e[0]["human_reset"]), key=lambda e: e[2]
                )
                # Direct labels for the no-human endpoints, which otherwise sit on top of
                # each other near the origin. The structured panel is empty at its top
                # left; the agentic panel is empty on its right above the flat tails.
                anchor_x, anchor_y = (
                    (0.04, 0.95) if panel_title == "Structured planners" else (0.20, 0.78)
                )
                for rank, (arm, x, y) in enumerate(shown):
                    text = f"{y:g}/10" if metric == "performance" else f"{y:g} hits"
                    status = "" if arm["complete"] else ", running"
                    label = f"{arm['method']}: {text} at {x:,.0f} steps{status}"
                    ax.annotate(
                        label,
                        xy=(x, y),
                        xytext=(anchor_x, anchor_y - 0.065 * rank),
                        textcoords="axes fraction",
                        fontsize=8,
                        color=FragileNoHumanReset.COLORS[arm["key"]],
                        bbox=dict(
                            boxstyle="round,pad=0.2",
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.85,
                        ),
                        arrowprops=dict(
                            arrowstyle="-",
                            linewidth=0.6,
                            color=FragileNoHumanReset.COLORS[arm["key"]],
                        ),
                        va="center",
                    )
                ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.17),
                    ncol=2 if len(keys) <= 3 else 1,
                    frameon=False,
                    fontsize=8,
                )
            fig.suptitle(
                f"Human reset removed from practice (dashed) vs. offered (solid) — {title}",
                x=0.01,
                ha="left",
                fontsize=12,
            )
            fig.text(
                0.01,
                0.005,
                "Tossing3D fragile: 100 kg movable bin, human cost 100, damage cost 10. Seed 0 "
                "unless n says otherwise. Diamonds mark actual endpoints; open circles mark runs "
                "still in progress at "
                f"{curves['snapshot']}.",
                fontsize=8,
                color="#555555",
            )
            fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.40, wspace=0.22)
            path = output_dir / f"{stem}-{metric}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)
        return written

    @staticmethod
    def main(*, argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--log-dir", type=Path, required=True)
        parser.add_argument("--artifacts-root", type=Path, default=None)
        parser.add_argument("--stem", default="2026-10-13-fragile-no-human-reset")
        args = parser.parse_args(argv)
        curves_path = args.log_dir / "curves.json"
        if args.artifacts_root is not None:
            curves = FragileNoHumanReset.extract(artifacts_root=args.artifacts_root)
            args.log_dir.mkdir(parents=True, exist_ok=True)
            curves_path.write_text(json.dumps(curves, indent=1) + "\n")
        else:
            curves = json.loads(curves_path.read_text())
        for path in FragileNoHumanReset.plot(
            curves=curves, output_dir=args.log_dir, stem=args.stem
        ):
            print(path)


if __name__ == "__main__":
    FragileNoHumanReset.main()
