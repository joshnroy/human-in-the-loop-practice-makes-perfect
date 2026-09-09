"""Compare exact expectimax with compute-bounded determinized A*.

Exact expectimax is first run to completion to establish its reference action,
value, evaluated-node count, and runtime. Both planners are then rerun through
the same budgeted interface: once with that exact node budget and once with the
same half-reference wall-clock deadline. Interrupted expectimax retains only
STOP and completely evaluated root actions; partial chance sums are discarded.
A* itself has no depth limit.
"""

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel

from hitl_pmp.methods.belief_space.determinized import DeterminizedAStarPlanner
from hitl_pmp.methods.belief_space.expectimax import ExpectimaxPlanner
from hitl_pmp.methods.belief_space.types.search_trace import SearchTrace


class BenchmarkState(BaseModel):
    model_config = {"frozen": True}
    path: str = ""


class BenchmarkBelief(BaseModel):
    model_config = {"frozen": True}
    value: float = 0.0


class BenchmarkTheta(BaseModel):
    value: float = 0.0


class BenchmarkAction(BaseModel):
    model_config = {"frozen": True}
    index: int


class BenchmarkModel:
    """Three actions, two chance outcomes, and a depth-six value peak."""

    actions = [BenchmarkAction(index=index) for index in range(3)]

    def sample_policy_values_from_belief(
        self, *, belief_state: BenchmarkBelief, num_samples: int
    ) -> np.ndarray:
        return np.full(num_samples, belief_state.value)

    def G(self, *, policy_value: float, summed_cost: float) -> float:
        return policy_value - 0.01 * summed_cost

    def get_valid_actions(self, *, environment_state: BenchmarkState) -> list[BenchmarkAction]:
        del environment_state
        return self.actions

    def transition_outcomes(
        self,
        *,
        environment_state: BenchmarkState,
        practice_action: BenchmarkAction,
        belief_state: BenchmarkBelief,
    ) -> list[tuple[BenchmarkState, float, float]]:
        del belief_state
        prefix = f"{environment_state.path}{practice_action.index}"
        return [
            (BenchmarkState(path=f"{prefix}L"), 1.0, 0.5),
            (BenchmarkState(path=f"{prefix}R"), 1.0, 0.5),
        ]

    def update_belief_state(
        self,
        *,
        belief_state: BenchmarkBelief,
        environment_state: BenchmarkState,
        potential_next_environment_state: BenchmarkState,
        practice_action: BenchmarkAction,
    ) -> BenchmarkBelief:
        del environment_state
        depth = len(potential_next_environment_state.path) // 2
        direction = 1.0 if depth <= 6 else -1.0
        outcome_bonus = 0.002 if potential_next_environment_state.path.endswith("R") else 0.0
        return BenchmarkBelief(
            value=belief_state.value
            + direction * (0.01 * (practice_action.index + 1) + outcome_bonus)
        )

    def search_cache_key(
        self,
        *,
        environment_state: BenchmarkState,
        summed_cost: float,
        belief_state: BenchmarkBelief,
        horizon: int,
    ) -> object:
        return environment_state, summed_cost, belief_state, horizon


def _summary(*, trace: SearchTrace) -> dict[str, Any]:
    return next(event for event in trace.events if event["event"] == "search_summary")


def _run(*, planner: Any, horizon: int) -> dict[str, Any]:
    trace = SearchTrace()
    value, action = planner.solve(
        environment_state=BenchmarkState(),
        summed_cost=0.0,
        belief_state=BenchmarkBelief(),
        horizon=horizon,
        model=BenchmarkModel(),
        num_samples=1,
        trace=trace,
    )
    summary = _summary(trace=trace)
    return {
        "elapsed_seconds": float(summary["elapsed_seconds"]),
        "expanded_nodes": int(summary["expanded_nodes"]),
        "evaluated_nodes": int(summary["evaluated_nodes"]),
        "generated_nodes": int(summary["generated_nodes"]),
        "max_depth_reached": int(summary["max_depth_reached"]),
        "termination_reason": str(summary["termination_reason"]),
        "time_budget_overshoot_seconds": float(summary.get("time_budget_overshoot_seconds") or 0.0),
        "value": float(value),
        "action": getattr(action, "index", "STOP"),
    }


def _aggregate(
    *, mode: str, planner: str, runs: list[dict[str, Any]], reference: dict[str, Any]
) -> dict[str, Any]:
    values = [float(run["value"]) for run in runs]
    elapsed = [float(run["elapsed_seconds"]) for run in runs]
    return {
        "budget_mode": mode,
        "planner": planner,
        "repeats": len(runs),
        "mean_seconds": statistics.mean(float(run["elapsed_seconds"]) for run in runs),
        "median_seconds": statistics.median(elapsed),
        "q1_seconds": float(np.quantile(elapsed, 0.25)),
        "q3_seconds": float(np.quantile(elapsed, 0.75)),
        "stdev_seconds": (
            statistics.stdev(float(run["elapsed_seconds"]) for run in runs)
            if len(runs) > 1
            else 0.0
        ),
        "mean_expanded_nodes": statistics.mean(int(run["expanded_nodes"]) for run in runs),
        "mean_evaluated_nodes": statistics.mean(int(run["evaluated_nodes"]) for run in runs),
        "mean_generated_nodes": statistics.mean(int(run["generated_nodes"]) for run in runs),
        "mean_max_depth": statistics.mean(int(run["max_depth_reached"]) for run in runs),
        "mean_time_budget_overshoot_seconds": statistics.mean(
            float(run["time_budget_overshoot_seconds"]) for run in runs
        ),
        "mean_value": statistics.mean(values),
        "mean_absolute_value_error": statistics.mean(
            abs(value - float(reference["value"])) for value in values
        ),
        "action_agreement_rate": statistics.mean(
            run["action"] == reference["action"] for run in runs
        ),
        "termination_reasons": ",".join(sorted({str(run["termination_reason"]) for run in runs})),
    }


def benchmark(*, reference_horizon: int, repeats: int, output: Path) -> list[dict[str, Any]]:
    exact_runs = [
        _run(
            planner=ExpectimaxPlanner[
                BenchmarkState, BenchmarkBelief, BenchmarkTheta, BenchmarkAction
            ](),
            horizon=reference_horizon,
        )
        for _ in range(repeats)
    ]
    reference = {
        "value": statistics.mean(float(run["value"]) for run in exact_runs),
        "action": exact_runs[0]["action"],
    }
    reference_nodes = round(statistics.mean(int(run["evaluated_nodes"]) for run in exact_runs))
    reference_seconds = statistics.median(float(run["elapsed_seconds"]) for run in exact_runs)
    wall_clock_budget = reference_seconds * 0.5
    paired_runs: dict[tuple[str, str], list[dict[str, Any]]] = {
        (mode, planner): []
        for mode in ("fixed_node_budget", "fixed_wall_clock_budget")
        for planner in ("expectimax", "determinized_astar")
    }

    def make_planner(*, mode: str, planner: str, repeat: int) -> Any:
        node_limit = reference_nodes if mode == "fixed_node_budget" else None
        time_limit = wall_clock_budget if mode == "fixed_wall_clock_budget" else None
        if planner == "expectimax":
            return ExpectimaxPlanner[
                BenchmarkState, BenchmarkBelief, BenchmarkTheta, BenchmarkAction
            ](max_evaluated_nodes=node_limit, max_seconds=time_limit)
        return DeterminizedAStarPlanner[
            BenchmarkState, BenchmarkBelief, BenchmarkTheta, BenchmarkAction
        ](max_evaluated_nodes=node_limit, max_seconds=time_limit, seed=repeat)

    # Alternate execution order within every paired repetition to reduce drift.
    for mode in ("fixed_node_budget", "fixed_wall_clock_budget"):
        for repeat in range(repeats):
            order = (
                ("expectimax", "determinized_astar")
                if repeat % 2 == 0
                else ("determinized_astar", "expectimax")
            )
            for planner in order:
                paired_runs[mode, planner].append(
                    _run(
                        planner=make_planner(mode=mode, planner=planner, repeat=repeat),
                        horizon=reference_horizon,
                    )
                )
    rows = []
    for (mode, planner), runs in paired_runs.items():
        rows.append({
            **_aggregate(mode=mode, planner=planner, runs=runs, reference=reference),
            "node_budget": reference_nodes if mode == "fixed_node_budget" else "",
            "time_budget_seconds": (wall_clock_budget if mode == "fixed_wall_clock_budget" else ""),
            "reference_horizon": reference_horizon,
            "reference_value": reference["value"],
            "reference_action": reference["action"],
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    node_rows = [row for row in rows if row["budget_mode"] == "fixed_node_budget"]
    time_rows = [row for row in rows if row["budget_mode"] == "fixed_wall_clock_budget"]
    axes[0].bar(
        [str(row["planner"]).replace("_", " ") for row in node_rows],
        [1000 * float(row["mean_seconds"]) for row in node_rows],
    )
    axes[0].set_ylabel("Wall-clock time (ms)")
    axes[0].set_title(f"Same node budget: {reference_nodes:,}")
    axes[1].bar(
        [str(row["planner"]).replace("_", " ") for row in time_rows],
        [float(row["mean_evaluated_nodes"]) for row in time_rows],
    )
    axes[1].set_ylabel("Evaluated nodes")
    axes[1].set_title(f"Same time budget: {1000 * wall_clock_budget:.1f} ms")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Compute-budgeted planner comparison")
    figure.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(figure)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-horizon", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for row in benchmark(
        reference_horizon=args.reference_horizon, repeats=args.repeats, output=args.output
    ):
        print(row)


if __name__ == "__main__":
    main()
