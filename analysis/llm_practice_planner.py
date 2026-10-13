"""Figures for the fragile Tossing3D LLM practice-planner log: actual endpoints, never filled.

Two modes. ``--artifacts-root`` reads every run directory listed in ``ARMS`` and writes
``curves.json`` into ``--log-dir``; without it the committed ``curves.json`` is re-plotted.
"""

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


class LlmPracticePlanner:
    # Colour carries the method (CLAUDE.md, "Training-curve style"). #CC79A7 and #E69F00
    # are taken by the two Full-Agentic arms in the fragile family, so LCCC takes the next
    # listed colour (#F0E442, darkened so a line reads on white) and LLCC a brown that no
    # figure in the family uses. Memory mode is a shade: stateless light, resume dark.
    COLORS = {
        "lccc-stateless": "#D9C81C",
        "lccc-resume": "#8A7F00",
        "llcc-stateless": "#B9854F",
        "llcc-resume": "#6B3A0E",
        "expectimax": "#0072B2",
    }
    MARKERS = {"stateless": "o", "resume": "s", None: None}
    BUDGET_COLOR = "#999999"
    # (key, method label, planner memory, path under artifacts/)
    ARMS: tuple[tuple[str, str, str | None, str], ...] = (
        (
            "lccc-stateless",
            "LCCC (LLM planner)",
            "stateless",
            "llm-planner-lccc-20261013/stateless/results/pomdp/0",
        ),
        (
            "lccc-resume",
            "LCCC (LLM planner)",
            "resume",
            "llm-planner-lccc-20261013/resume/results/pomdp/0",
        ),
        (
            "llcc-stateless",
            "LLCC (LLM planner + LLM estimator)",
            "stateless",
            "llm-planner-llcc-20261013/stateless/results/pomdp/0",
        ),
        (
            "llcc-resume",
            "LLCC (LLM planner + LLM estimator)",
            "resume",
            "llm-planner-llcc-20261013/resume/results/pomdp/0",
        ),
        (
            "expectimax",
            "POMDP Planner / expectimax",
            None,
            "fragile-mat-comparison-notion-20261009/report/expectimax",
        ),
    )
    REASON_PATTERNS = {
        "practice": r"practi",
        "step_budget": r"budget",
        "actions_left": r"actions? (?:remain|left)",
        "cost": r"cost|cheap|\b100\b",
        "human": r"human",
    }

    @staticmethod
    def score_summary(*, evaluations: list[list[int]]) -> dict[str, Any]:
        solved = [row[1] for row in evaluations]
        return dict(
            final=solved[-1],
            best=max(solved),
            mean_last_three=sum(solved[-3:]) / len(solved[-3:]),
            largest_swing=max(abs(b - a) for a, b in zip(solved, solved[1:], strict=False)),
        )

    @staticmethod
    def _close(*, events: list[list[Any]], endpoint: int) -> list[list[Any]]:
        if events[-1][0] > endpoint:
            raise ValueError(f"event at step {events[-1][0]} is past the endpoint {endpoint}")
        if events[-1][0] != endpoint:
            events.append([endpoint, *events[-1][1:]])
        return events

    @staticmethod
    def _action_name(*, action: Any) -> str:
        return action if isinstance(action, str) else action["skill"]["name"]

    @staticmethod
    def _decisions(*, root: Path, endpoint: int, spent: float) -> dict[str, Any]:
        spend = [[0, 0.0]]
        costs, resets = [], []
        stops = 0
        opens_session = False
        final = None
        with (root / "pomdp_decisions.jsonl").open() as stream:
            for line in stream:
                record = json.loads(line)
                if record["event"] == "session_start":
                    opens_session = True
                    continue
                if record["event"] != "decision":
                    continue
                name = LlmPracticePlanner._action_name(action=record["action"])
                cost = float(record["llm_cost_usd"])
                costs.append(cost)
                spend.append([int(record["practice_steps_used"]), spend[-1][1] + cost])
                stops += name == "STOP"
                if name.startswith("ask_for_reset"):
                    resets.append(record.get("llm_reason") or "")
                final = dict(
                    action=name,
                    first_in_session=opens_session,
                    practice_steps=int(record["practice_steps_used"]),
                    reason=record.get("llm_reason"),
                )
                opens_session = False
        if spend[-1][1] > spent + 1e-9:
            raise ValueError(f"decisions cost {spend[-1][1]:.4f}, the ledger says {spent:.4f}")
        if spent > spend[-1][1] + 1e-9:
            # A call the ledger charged that produced no decision (one that hit the budget).
            spend.append([endpoint, spent])
        ordered = sorted(costs)
        reasons = {
            key: sum(1 for r in resets if re.search(pattern, r, re.IGNORECASE))
            for key, pattern in LlmPracticePlanner.REASON_PATTERNS.items()
        }
        return dict(
            spend_events=LlmPracticePlanner._close(events=spend, endpoint=endpoint),
            decisions=len(costs),
            stop_decisions=stops,
            final_decision=final,
            reset_reasons=dict(total=len(resets), **reasons),
            decision_cost_usd=dict(
                median=statistics.median(costs),
                p10=ordered[len(ordered) // 10],
                p90=ordered[(9 * len(ordered)) // 10],
                min=ordered[0],
                max=ordered[-1],
            ),
        )

    @staticmethod
    def load_run(*, root: Path) -> dict[str, Any]:
        status = json.loads((root / "status.json").read_text())
        endpoint = int(status["practice_steps"])
        evaluations = [
            [r["practice_steps"], r["num_solved"], r["num_total"]]
            for r in json.loads((root / "step_evaluations.json").read_text())
            if r.get("complete", True)
        ]
        humans, damage = [[0, 0, 0]], [[0, 0]]
        endpoint_reason = None
        with (root / "step_events.jsonl").open() as stream:
            for line in stream:
                event = json.loads(line)
                if event["event"] == "human":
                    near, far = humans[-1][1:]
                    near += event["destination"] == "robot_side"
                    far += event["destination"] == "opposite_side"
                    humans.append([int(event["end_step"]), near, far])
                elif event["event"] == "damage":
                    damage.append([event["practice_step"], damage[-1][1] + len(event["impacts"])])
                elif event["event"] == "practice_end":
                    endpoint_reason = event["reason"]
        if damage[-1][1] != int(status["damage_contacts"]):
            raise ValueError(f"damage events sum to {damage[-1][1]}, status says otherwise")
        sides = status.get("human_resets_by_destination")
        if sides is not None and humans[-1][1:] != [sides["robot_side"], sides["opposite_side"]]:
            raise ValueError(f"human events {humans[-1][1:]} disagree with status {sides}")
        run = dict(
            evaluations=evaluations,
            human_events=LlmPracticePlanner._close(events=humans, endpoint=endpoint),
            damage_events=LlmPracticePlanner._close(events=damage, endpoint=endpoint),
            endpoint_steps=endpoint,
            endpoint_reason=endpoint_reason,
            physical_cost=status.get("physical_cost"),
            score=LlmPracticePlanner.score_summary(evaluations=evaluations),
        )
        ledger = root / "model_budget.json"
        if ledger.exists():
            spent = float(json.loads(ledger.read_text())["spent"])
            run.update(
                model_spent_usd=spent,
                model_calls=len(list((root / "planner_calls").glob("*/result.json"))),
                **LlmPracticePlanner._decisions(root=root, endpoint=endpoint, spent=spent),
            )
        return run

    @staticmethod
    def extract(*, artifacts_root: Path) -> dict[str, Any]:
        arms = []
        for key, method, memory, relative in LlmPracticePlanner.ARMS:
            run = LlmPracticePlanner.load_run(root=artifacts_root / relative)
            arms.append(dict(key=key, method=method, memory=memory, seed=0, **run))
        return dict(arms=arms)

    @staticmethod
    def footer() -> str:
        return (
            "Tossing3D fragile: 1 m mat, 100 kg movable bin, human cost 100, damage cost 10, "
            "human reset available.\nSeed 0, n=1 per arm. Diamonds mark actual endpoints."
        )

    @staticmethod
    def _label(*, arm: dict[str, Any], detail: str) -> str:
        memory = f" / {arm['memory']}" if arm["memory"] else ""
        return f"{arm['method']}{memory} — n=1, {detail}"

    @staticmethod
    def _draw(*, ax: Any, arm: dict[str, Any], x: list[Any], y: list[Any], label: str) -> None:
        colour = LlmPracticePlanner.COLORS[arm["key"]]
        step = {"drawstyle": "steps-post"}
        ax.plot(x, y, color=colour, linewidth=0.8, alpha=0.16, **step)
        ax.plot(x, y, color=colour, linewidth=2.3, label=label, **step)
        ax.plot([x[-1]], [y[-1]], color=colour, linestyle="none", marker="D", markersize=6)

    @staticmethod
    def _axes(*, ax: Any, ylabel: str) -> None:
        from matplotlib.ticker import FuncFormatter

        ax.set_xlim(0, 88000)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xlabel("counted practice steps")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)

    @staticmethod
    def _figure(*, plt: Any, title: str, panels: int = 1) -> tuple[Any, Any]:
        fig, axes = plt.subplots(1, panels, figsize=(7.4 * panels, 6.4), squeeze=False)
        fig.suptitle(title, x=0.01, ha="left", fontsize=12)
        fig.text(0.01, 0.01, LlmPracticePlanner.footer(), fontsize=8, color="#555555")
        return fig, axes[0]

    @staticmethod
    def _legend(*, ax: Any) -> None:
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, frameon=False, fontsize=8
        )

    @staticmethod
    def plot(*, curves: dict[str, Any], output_dir: Path, stem: str) -> list[Path]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
        output_dir.mkdir(parents=True, exist_ok=True)
        arms = curves["arms"]
        llm = [a for a in arms if "spend_events" in a]
        written = []

        def save(*, fig: Any, name: str) -> None:
            fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.36, wspace=0.22)
            path = output_dir / f"{stem}-{name}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)

        fig, (ax,) = LlmPracticePlanner._figure(
            plt=plt, title="Held-out tasks solved (of 10) vs. counted practice steps"
        )
        for arm in arms:
            x = [r[0] for r in arm["evaluations"]]
            y = [r[1] for r in arm["evaluations"]]
            colour = LlmPracticePlanner.COLORS[arm["key"]]
            score = arm["score"]
            label = LlmPracticePlanner._label(
                arm=arm,
                detail=f"final {score['final']}/10, best {score['best']}/10, "
                f"ends {arm['endpoint_steps']:,} ({arm['endpoint_reason']})",
            )
            ax.plot(x, y, color=colour, linewidth=0.8, alpha=0.16)
            ax.plot(
                x,
                y,
                color=colour,
                linewidth=2.3,
                marker=LlmPracticePlanner.MARKERS[arm["memory"]],
                markersize=3.5,
                label=label,
            )
            ax.plot([x[-1]], [y[-1]], color=colour, linestyle="none", marker="D", markersize=7)
        LlmPracticePlanner._axes(ax=ax, ylabel="solved per seed")
        ax.set_ylim(-0.4, 10.6)
        ax.set_yticks(range(0, 11, 2))
        ax.set_title("Tossing3D fragile, 1 m mat", loc="left", fontsize=10)
        LlmPracticePlanner._legend(ax=ax)
        save(fig=fig, name="performance")

        fig, axes = LlmPracticePlanner._figure(
            plt=plt, title="Cumulative human resets vs. counted practice steps, by side", panels=2
        )
        for ax, column, side in zip(axes, (1, 2), ("near (robot) side", "far side"), strict=True):
            for arm in arms:
                x = [r[0] for r in arm["human_events"]]
                y = [r[column] for r in arm["human_events"]]
                LlmPracticePlanner._draw(
                    ax=ax,
                    arm=arm,
                    x=x,
                    y=y,
                    label=LlmPracticePlanner._label(arm=arm, detail=f"{y[-1]} {side}"),
                )
            LlmPracticePlanner._axes(ax=ax, ylabel="cumulative human resets per seed")
            ax.set_title(f"Resets to the {side} — Tossing3D fragile", loc="left", fontsize=10)
            LlmPracticePlanner._legend(ax=ax)
        save(fig=fig, name="human-resets")

        fig, (ax,) = LlmPracticePlanner._figure(
            plt=plt, title="Cumulative model spend (practice ledger) vs. counted practice steps"
        )
        ax.axhline(
            20.0,
            color=LlmPracticePlanner.BUDGET_COLOR,
            linestyle=":",
            linewidth=1.5,
            label="$20 practice model budget (reference)",
        )
        for arm in llm:
            x = [r[0] for r in arm["spend_events"]]
            y = [r[1] for r in arm["spend_events"]]
            LlmPracticePlanner._draw(
                ax=ax,
                arm=arm,
                x=x,
                y=y,
                label=LlmPracticePlanner._label(
                    arm=arm, detail=f"${y[-1]:.2f} over {arm['model_calls']} calls"
                ),
            )
        LlmPracticePlanner._axes(ax=ax, ylabel="cumulative model spend per seed (USD)")
        ax.set_xlim(0, 37000)
        ax.set_ylim(0, 21.5)
        ax.set_title("Tossing3D fragile, 1 m mat (LLM arms only)", loc="left", fontsize=10)
        LlmPracticePlanner._legend(ax=ax)
        save(fig=fig, name="model-spend")

        fig, (ax,) = LlmPracticePlanner._figure(
            plt=plt, title="Cumulative raw-ground hits vs. counted practice steps"
        )
        for arm in arms:
            x = [r[0] for r in arm["damage_events"]]
            y = [r[1] for r in arm["damage_events"]]
            LlmPracticePlanner._draw(
                ax=ax,
                arm=arm,
                x=x,
                y=y,
                label=LlmPracticePlanner._label(arm=arm, detail=f"{y[-1]} hits"),
            )
        LlmPracticePlanner._axes(ax=ax, ylabel="cumulative raw-ground hits per seed")
        ax.set_title("Tossing3D fragile, 1 m mat", loc="left", fontsize=10)
        LlmPracticePlanner._legend(ax=ax)
        save(fig=fig, name="raw-ground-hits")
        return written

    @staticmethod
    def main(*, argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--log-dir", type=Path, required=True)
        parser.add_argument("--artifacts-root", type=Path, default=None)
        parser.add_argument("--stem", default="2026-10-13-llm-practice-planner")
        args = parser.parse_args(argv)
        data = args.log_dir / "curves.json"
        if args.artifacts_root is not None:
            curves = LlmPracticePlanner.extract(artifacts_root=args.artifacts_root)
            args.log_dir.mkdir(parents=True, exist_ok=True)
            data.write_text(json.dumps(curves, indent=1) + "\n")
        curves = json.loads(data.read_text())
        for path in LlmPracticePlanner.plot(curves=curves, output_dir=args.log_dir, stem=args.stem):
            print(path)


if __name__ == "__main__":
    LlmPracticePlanner.main()
