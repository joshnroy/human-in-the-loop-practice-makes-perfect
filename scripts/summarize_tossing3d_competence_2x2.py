"""Validate completed 2x2 runs and export compact summaries and comparison figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ARMS = ("global_curve-particle", "global_curve-grid", "local_trend-particle", "local_trend-grid")
TOSS_SKILL = "MoveToTossLocationAndToss"
RESET_SKILLS = ("ask_for_reset_cube_bin_only", "non_human_reset_cube_bin_only")


def read_json(*, path: Path) -> Any:
    """Only the small manifest/status/stats files use this helper."""
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def iter_jsonl(*, path: Path) -> Iterator[dict[str, Any]]:
    """Keep at most one log record in memory, including large grid-belief records."""
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield record


def scan_decisions(*, path: Path, num_cycles: int) -> dict[str, Any]:
    event_cycles: dict[str, list[int]] = defaultdict(list)
    dispatch_count: Counter[str] = Counter()
    dispatch_cost: dict[str, float] = defaultdict(float)
    costs = [0.0] * num_cycles
    counts = [0] * num_cycles
    training: dict[int, dict[str, int]] = {}
    outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    final_history: dict[str, list[dict[str, Any]]] = {}
    latest_diagnostics: dict[str, Any] = {}
    first_filtered: dict[str, Any] = {}
    final_filtered: dict[str, Any] = {}
    final_predicted: dict[str, Any] = {}
    errors = []
    for record in iter_jsonl(path=path):
        event = record.get("event")
        cycle = int(record.get("cycle", -1))
        if event in ("dispatch", "outcome", "refit", "smoothing") and not 0 <= cycle < num_cycles:
            errors.append(f"{event} has cycle {cycle}, outside 0..{num_cycles - 1}")
            continue
        if event == "dispatch":
            skill, cost = str(record["skill"]), float(record["cost"])
            if not math.isfinite(cost) or cost < 0:
                errors.append(f"dispatch cost is invalid: {cost}")
                continue
            dispatch_count[skill] += 1
            dispatch_cost[skill] += cost
            counts[cycle] += 1
            costs[cycle] += cost
            if not math.isclose(float(record["summed_cost"]), costs[cycle], abs_tol=1e-8):
                errors.append(f"cycle {cycle}: dispatch summed_cost disagrees with event costs")
        elif event == "outcome":
            tally = outcomes[str(record["skill"])]
            tally["attempts"] += 1
            tally["successes"] += int(record["success"])
            tally["random_attempts"] += int(record.get("random_exploration", False))
            if not record.get("random_exploration", False):
                tally["inference_attempts"] += 1
                tally["inference_successes"] += int(record["success"])
            latest_diagnostics = record.get("beliefs", latest_diagnostics)
        elif event == "smoothing":
            event_cycles["smoothing"].append(cycle)
            if record.get("use") != "retrospective_diagnostics_only":
                errors.append(f"cycle {cycle}: smoothing use is not retrospective diagnostics")
            final_history = record["history"]
            for skill, history in final_history.items():
                if [row["cycle_index"] for row in history] != list(range(cycle + 1)):
                    errors.append(f"cycle {cycle}: nonconsecutive smoothing history for {skill}")
            final_filtered = latest_diagnostics
            if not first_filtered:
                first_filtered = latest_diagnostics
        elif event == "refit":
            event_cycles["refit"].append(cycle)
            if record.get("learning_rate_evidence") != "success_failure_only":
                errors.append(
                    f"cycle {cycle}: refit does not report S/F-only learning-rate evidence"
                )
            training[cycle] = record.get("training_examples", {})
            final_predicted = record.get("beliefs", {})
            latest_diagnostics = final_predicted
        # In particular, never retain record['belief'], which embeds all array buffers.
    for event in ("refit", "smoothing"):
        if event_cycles[event] != list(range(num_cycles)):
            errors.append(
                f"expected {num_cycles} consecutive {event} events; got {event_cycles[event]}"
            )
    if TOSS_SKILL not in final_history:
        errors.append("missing final toss smoothing history")
    ancestry = {}
    for skill, history in final_history.items():
        if len(history) != num_cycles:
            errors.append(
                f"final {skill} smoothing history has {len(history)} cycles, expected {num_cycles}"
            )
        if not history:
            continue
        last = history[-1]
        for suffix in ("competence", "learning_rate"):
            if not math.isclose(
                last[f"filtered_{suffix}"], last[f"smoothed_{suffix}"], abs_tol=1e-10
            ):
                errors.append(f"terminal filtered/smoothed {skill} {suffix} disagree")
        ancestry[skill] = {
            "earliest_smoothed_ess": history[0]["effective_sample_size"],
            "final_filtered_ess": last["effective_sample_size"],
            "earliest_unique_ancestors": history[0].get("unique_ancestors"),
            "final_unique_ancestors": last.get("unique_ancestors"),
            "phi_configurations_after_first_cycle": first_filtered.get(skill, {}).get(
                "unique_phi_configurations"
            ),
            "phi_configurations_final_filtered": final_filtered.get(skill, {}).get(
                "unique_phi_configurations"
            ),
            "phi_configurations_after_final_refit": final_predicted.get(skill, {}).get(
                "unique_phi_configurations"
            ),
        }
    return {
        "errors": errors,
        "refit_count": len(event_cycles["refit"]),
        "smoothing_count": len(event_cycles["smoothing"]),
        "dispatch_by_skill": {
            skill: {"count": count, "cost": dispatch_cost[skill]}
            for skill, count in sorted(dispatch_count.items())
        },
        "dispatched_count_by_cycle": counts,
        "dispatched_cost_by_cycle": costs,
        "training_examples_by_cycle": [training.get(cycle, {}) for cycle in range(num_cycles)],
        "outcomes_by_skill": {skill: dict(tally) for skill, tally in sorted(outcomes.items())},
        "toss_history": final_history.get(TOSS_SKILL, []),
        "ancestry_diagnostics": ancestry,
        "final_filtered_diagnostics": final_filtered,
        "final_predicted_diagnostics": final_predicted,
    }


def scan_sampler(*, path: Path, num_cycles: int) -> dict[str, Any]:
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    by_cycle = [{"attempts": 0, "successes": 0} for _ in range(num_cycles)]
    consultations: Counter[str] = Counter()
    errors = []
    if path.exists():
        for row in iter_jsonl(path=path):
            cycle = int(row["cycle"])
            if not 0 <= cycle < num_cycles:
                errors.append(f"sampler draw has cycle {cycle}, outside 0..{num_cycles - 1}")
                continue
            skill, success = str(row["skill"]), int(row["success"])
            totals[skill]["attempts"] += 1
            totals[skill]["successes"] += success
            by_cycle[cycle]["attempts"] += 1
            by_cycle[cycle]["successes"] += success
            consultations[str(row["consultation"])] += 1
    return {
        "errors": errors,
        "sampler_by_skill": {skill: dict(tally) for skill, tally in sorted(totals.items())},
        "sampler_by_cycle": by_cycle,
        "sampler_consultations": dict(sorted(consultations.items())),
    }


def summarize_run(
    *, root: Path, name: str, num_cycles: int, status: dict[str, Any] | None
) -> dict[str, Any]:
    directory = root / name
    errors = []
    arm, seed_label = name.split("/")
    model, engine = arm.rsplit("-", 1)
    result: dict[str, Any] = {
        "name": name,
        "model": model,
        "engine": engine,
        "seed": int(seed_label.removeprefix("seed_")),
        "directory": str(directory),
        "returncode": None if status is None else status.get("returncode"),
    }
    if status is None or status.get("returncode") != 0:
        errors.append(f"missing successful run status (returncode={result['returncode']})")
    if not (directory / "stats.json").exists():
        return {**result, "valid": False, "errors": [*errors, "missing stats.json"]}
    stats = read_json(path=directory / "stats.json")
    evaluations = stats["evaluations"]
    sessions = stats.get("practice_session_ends", [])
    if len(evaluations) != num_cycles + 1:
        errors.append(f"expected {num_cycles + 1} evaluation sweeps, got {len(evaluations)}")
    if [session["cycle_index"] for session in sessions] != list(range(num_cycles)):
        errors.append(f"expected {num_cycles} consecutive practice_session_ends")
    progress = list(iter_jsonl(path=directory / "progress.jsonl"))
    if len(progress) != num_cycles + 1:
        errors.append(f"expected {num_cycles + 1} progress sweeps, got {len(progress)}")
    for index, row in enumerate(progress):
        if row["sweeps_completed"] != index + 1 or row["sweeps_total"] != num_cycles + 1:
            errors.append(f"progress sweep {index}: incorrect completed/total counters")
        if (
            index < len(evaluations)
            and [row["transitions"], row["num_solved"], row["num_total"]] != evaluations[index]
        ):
            errors.append(f"progress sweep {index} disagrees with stats.json")
    decisions = scan_decisions(path=directory / "pomdp_decisions.jsonl", num_cycles=num_cycles)
    sampler = scan_sampler(path=directory / "sampler_draws.jsonl", num_cycles=num_cycles)
    errors.extend(decisions.pop("errors"))
    errors.extend(sampler.pop("errors"))
    for cycle, session in enumerate(sessions[:num_cycles]):
        if decisions["dispatched_count_by_cycle"][cycle] != session["actions_executed"]:
            errors.append(f"cycle {cycle}: dispatch count disagrees with actions_executed")
    free_resets = int(stats.get("num_practice_resets", 0))
    if free_resets:
        errors.append(f"observed {free_resets} free practice resets in a reset-free protocol")
    expected_sampler_attempts = sum(
        tally["num_attempts"] - tally["num_unparameterized_attempts"]
        for window in stats.get("practice_outcomes_per_cycle", [])
        for tally in window.values()
    )
    sampler_attempts = sum(row["attempts"] for row in sampler["sampler_by_cycle"])
    if sampler_attempts != expected_sampler_attempts:
        errors.append(
            f"sampler log has {sampler_attempts} draws, stats require {expected_sampler_attempts}"
        )
    cumulative = 0.0
    cycles = []
    for index in range(num_cycles):
        cost = decisions["dispatched_cost_by_cycle"][index]
        cumulative += cost
        cycles.append({
            "cycle": index + 1,
            "actions_dispatched": decisions["dispatched_count_by_cycle"][index],
            "dispatched_cost": cost,
            "cumulative_dispatched_cost": cumulative,
            "session_end_reason": sessions[index]["reason"] if index < len(sessions) else None,
            "training_examples": decisions["training_examples_by_cycle"][index],
            "sampler_attempts": sampler["sampler_by_cycle"][index]["attempts"],
            "sampler_successes": sampler["sampler_by_cycle"][index]["successes"],
        })
    reset_dispatches = {
        skill: decisions["dispatch_by_skill"].get(skill, {"count": 0, "cost": 0.0})
        for skill in RESET_SKILLS
    }
    recorded_reset_count = int(stats.get("num_human_interventions_recorded", 0))
    recorded_reset_cost = float(stats.get("summed_human_cost_recorded", 0.0))
    if recorded_reset_count != sum(row["count"] for row in reset_dispatches.values()):
        errors.append("reset dispatch counts disagree with recorded intervention count")
    if not math.isclose(
        recorded_reset_cost, sum(row["cost"] for row in reset_dispatches.values()), abs_tol=1e-8
    ):
        errors.append("reset dispatch costs disagree with recorded intervention cost")
    return {
        **result,
        **decisions,
        **sampler,
        "valid": not errors,
        "errors": errors,
        "num_cycles": num_cycles,
        "practice_session_count": len(sessions),
        "progress_count": len(progress),
        "evaluations": [
            {
                "cycle": index,
                "transitions": row[0],
                "solved": row[1],
                "total": row[2],
                "success_rate": row[1] / row[2] if row[2] else None,
            }
            for index, row in enumerate(evaluations)
        ],
        "cycles": cycles,
        "total_dispatched_cost": cumulative,
        "total_dispatches": sum(decisions["dispatched_count_by_cycle"]),
        "free_practice_resets": free_resets,
        "reset_dispatches": reset_dispatches,
        "recorded_intervention_count": recorded_reset_count,
        "recorded_intervention_cost": recorded_reset_cost,
        "elapsed_seconds": progress[-1].get("elapsed_seconds") if progress else None,
    }


def summarize_results(*, results_root: Path, num_cycles: int | None = None) -> dict[str, Any]:
    root = results_root.resolve()
    manifest_path = root / "manifest.json"
    manifest = read_json(path=manifest_path) if manifest_path.exists() else {}
    if num_cycles is None:
        num_cycles = manifest.get("num_cycles")
    elif manifest and num_cycles != manifest["num_cycles"]:
        raise ValueError("--num-cycles disagrees with manifest.json")
    if num_cycles is None or num_cycles < 1:
        raise ValueError("a manifest with num_cycles or explicit --num-cycles is required")
    if manifest:
        names = sorted(manifest["commands"])
        seeds = int(manifest["num_seeds"])
        expected = {f"{arm}/seed_{seed:02d}" for arm in ARMS for seed in range(seeds)}
        if set(names) != expected:
            raise ValueError(
                "manifest commands do not contain the complete model/engine/seed product"
            )
    else:
        names = sorted(
            str(path.relative_to(root)) for arm in ARMS for path in (root / arm).glob("seed_*")
        )
        seed_labels = {name.split("/")[1] for name in names}
        if not seed_labels or set(names) != {
            f"{arm}/{seed}" for arm in ARMS for seed in seed_labels
        }:
            raise ValueError("run tree does not contain all four arms with the same seeds")
    statuses: dict[str, dict[str, Any]] = {}
    status_path = root / "run_status.json"
    if status_path.exists():
        for status in read_json(path=status_path):
            name = status["name"]
            if name in statuses:
                raise ValueError(f"duplicate run status: {name}")
            statuses[name] = status
    for name in names:
        individual = root / "logs" / (name.replace("/", "-") + ".status.json")
        if name not in statuses and individual.exists():
            statuses[name] = read_json(path=individual)
    runs = []
    for name in names:
        try:
            runs.append(
                summarize_run(
                    root=root, name=name, num_cycles=num_cycles, status=statuses.get(name)
                )
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            runs.append({"name": name, "valid": False, "errors": [str(exc)]})
    errors = [f"{run['name']}: {error}" for run in runs for error in run["errors"]]
    return {
        "schema_version": 1,
        "results_root": str(root),
        "source_commit": manifest.get("source_commit"),
        "manifest_present": bool(manifest),
        "num_cycles": num_cycles,
        "num_runs": len(runs),
        "valid": not errors,
        "errors": errors,
        "runs": runs,
        "notes": [
            "Dispatch costs include attempted actions under the preserved cost contract.",
            "Filtered/smoothed competence records are cycle-end values before each refit.",
            "Final smoothing uses all recorded cycles and never changes online decisions.",
            "The legacy intervention counter includes both named reset mechanisms; "
            "their dispatches are separated.",
        ],
    }


def write_summary(*, summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    columns = [
        "name",
        "model",
        "engine",
        "seed",
        "valid",
        "returncode",
        "num_cycles",
        "initial_eval_solved",
        "final_eval_solved",
        "eval_tasks",
        "total_dispatches",
        "total_dispatched_cost",
        "free_practice_resets",
        "sampler_attempts",
        "sampler_successes",
        "toss_final_filtered_competence",
        "toss_final_filtered_learning_rate",
        "toss_earliest_smoothed_ess",
        "toss_earliest_unique_ancestors",
        "toss_phi_configurations_final_filtered",
        "elapsed_seconds",
        "errors",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for run in summary["runs"]:
            row = {key: run.get(key) for key in columns}
            evaluations, history = run.get("evaluations", []), run.get("toss_history", [])
            ancestry = run.get("ancestry_diagnostics", {}).get(TOSS_SKILL, {})
            row.update({
                "initial_eval_solved": evaluations[0]["solved"] if evaluations else None,
                "final_eval_solved": evaluations[-1]["solved"] if evaluations else None,
                "eval_tasks": evaluations[-1]["total"] if evaluations else None,
                "sampler_attempts": sum(
                    tally["attempts"] for tally in run.get("sampler_by_cycle", [])
                ),
                "sampler_successes": sum(
                    tally["successes"] for tally in run.get("sampler_by_cycle", [])
                ),
                "toss_final_filtered_competence": history[-1]["filtered_competence"]
                if history
                else None,
                "toss_final_filtered_learning_rate": history[-1]["filtered_learning_rate"]
                if history
                else None,
                "toss_earliest_smoothed_ess": ancestry.get("earliest_smoothed_ess"),
                "toss_earliest_unique_ancestors": ancestry.get("earliest_unique_ancestors"),
                "toss_phi_configurations_final_filtered": ancestry.get(
                    "phi_configurations_final_filtered"
                ),
                "errors": "; ".join(run["errors"]),
            })
            writer.writerow(row)


def plot_summary(*, summary: dict[str, Any], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not summary["valid"]:
        raise ValueError("refusing to plot an incomplete or inconsistent experiment")
    colors = dict(zip(ARMS, ("#0072B2", "#D55E00", "#009E73", "#CC79A7"), strict=True))
    titles = {"global_curve": "A · global curve", "local_trend": "B · local trend"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.6), layout="constrained")
    for run in summary["runs"]:
        color = colors[f"{run['model']}-{run['engine']}"]
        label = f"{titles[run['model']]} / {run['engine']}"
        if summary["num_runs"] > 4:
            label += f" / seed {run['seed']}"
        evaluations, cycles, history = run["evaluations"], run["cycles"], run["toss_history"]
        axes[0, 0].plot(
            [row["cycle"] for row in evaluations],
            [row["success_rate"] for row in evaluations],
            "o-",
            color=color,
            label=label,
            markersize=4,
        )
        axes[0, 1].plot(
            [0, *[row["cycle"] for row in cycles]],
            [0, *[row["cumulative_dispatched_cost"] for row in cycles]],
            "o-",
            color=color,
            markersize=4,
        )
        for axis, suffix in ((axes[1, 0], "competence"), (axes[1, 1], "learning_rate")):
            x = [row["cycle_index"] + 1 for row in history]
            axis.plot(
                x,
                [row[f"filtered_{suffix}"] for row in history],
                ":o",
                color=color,
                alpha=0.65,
                markerfacecolor="white",
                markersize=4,
            )
            axis.plot(
                x, [row[f"smoothed_{suffix}"] for row in history], "-", color=color, linewidth=2
            )
    axes[0, 0].set(
        title="Held-out evaluation", ylabel="Tasks solved / tasks evaluated", ylim=(-0.03, 1.03)
    )
    axes[0, 1].set(
        title="Cumulative dispatched cost", ylabel="Configured practice cost", ylim=(0, None)
    )
    axes[1, 0].set(
        title="Toss competence: filtered and final smoothing",
        ylabel="Expected competence",
        ylim=(-0.03, 1.03),
    )
    axes[1, 1].set(
        title="Toss learning rate: filtered and final smoothing",
        ylabel="Expected competence gain / example",
        ylim=(0, None),
    )
    for axis in axes.ravel():
        axis.set_xlabel("Practice cycle")
        axis.set_xticks(range(summary["num_cycles"] + 1))
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    axes[1, 0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#555555",
                linestyle=":",
                marker="o",
                markerfacecolor="white",
                label="Online filtered",
            ),
            Line2D([0], [0], color="#555555", linewidth=2, label="Smoothed using all cycles"),
        ],
        fontsize=8,
        loc="upper left",
    )
    figure.suptitle(
        f"Tossing3D competence experiment · {summary['num_cycles']} practice cycles", fontsize=15
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "comparison.png", dpi=180)
    figure.savefig(output_dir / "comparison.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--num-cycles", type=int, help="Required only for a manifestless smoke tree"
    )
    args = parser.parse_args()
    try:
        summary = summarize_results(results_root=args.results_root, num_cycles=args.num_cycles)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    output = args.output_dir or args.results_root
    write_summary(summary=summary, output_dir=output)
    if not summary["valid"]:
        print("\n".join(summary["errors"]), file=sys.stderr)
        raise SystemExit(1)
    plot_summary(summary=summary, output_dir=output)
    print(
        f"Validated {summary['num_runs']} runs × {summary['num_cycles']} cycles; outputs: {output}"
    )


if __name__ == "__main__":
    main()
