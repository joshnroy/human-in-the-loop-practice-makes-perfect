"""Compare exact expectimax with compute-bounded determinized A*.

Exact expectimax is not an anytime algorithm: interrupting its recursive chance
sum does not yield an unbiased action value. We therefore run it to completion
at a reference horizon, then give A* either the same node or elapsed-time budget.
A* itself has no depth limit.
"""

import argparse
import csv
import statistics
import time
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
    started = time.perf_counter()
    value, action = planner.solve(
        environment_state=BenchmarkState(),
        summed_cost=0.0,
        belief_state=BenchmarkBelief(),
        horizon=horizon,
        model=BenchmarkModel(),
        num_samples=1,
        trace=trace,
    )
    elapsed = time.perf_counter() - started
    summary = _summary(trace=trace)
    return {
        "elapsed_seconds": elapsed,
        "expanded_nodes": int(summary["expanded_nodes"]),
        "generated_nodes": int(summary["generated_nodes"]),
        "max_depth_reached": int(summary["max_depth_reached"]),
        "termination_reason": str(summary["termination_reason"]),
        "value": float(value),
        "action": getattr(action, "index", "STOP"),
    }


def _aggregate(
    *, mode: str, planner: str, runs: list[dict[str, Any]], reference: dict[str, Any]
) -> dict[str, Any]:
    values = [float(run["value"]) for run in runs]
    return {
        "budget_mode": mode,
        "planner": planner,
        "repeats": len(runs),
        "mean_seconds": statistics.mean(float(run["elapsed_seconds"]) for run in runs),
        "stdev_seconds": (
            statistics.stdev(float(run["elapsed_seconds"]) for run in runs)
            if len(runs) > 1
            else 0.0
        ),
        "mean_expanded_nodes": statistics.mean(int(run["expanded_nodes"]) for run in runs),
        "mean_generated_nodes": statistics.mean(int(run["generated_nodes"]) for run in runs),
        "mean_max_depth": statistics.mean(int(run["max_depth_reached"]) for run in runs),
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
    node_budget = round(statistics.mean(int(run["expanded_nodes"]) for run in exact_runs))
    time_budget = statistics.median(float(run["elapsed_seconds"]) for run in exact_runs)
    node_runs = [
        _run(
            planner=DeterminizedAStarPlanner[
                BenchmarkState, BenchmarkBelief, BenchmarkTheta, BenchmarkAction
            ](max_expansions=node_budget, seed=repeat),
            horizon=reference_horizon,
        )
        for repeat in range(repeats)
    ]
    time_runs = [
        _run(
            planner=DeterminizedAStarPlanner[
                BenchmarkState, BenchmarkBelief, BenchmarkTheta, BenchmarkAction
            ](max_seconds=time_budget, seed=repeat),
            horizon=reference_horizon,
        )
        for repeat in range(repeats)
    ]
    common = {
        "node_budget": node_budget,
        "time_budget_seconds": time_budget,
        "reference_horizon": reference_horizon,
    }
    rows = [
        {
            **_aggregate(
                mode="reference_complete",
                planner="expectimax",
                runs=exact_runs,
                reference=reference,
            ),
            **common,
        },
        {
            **_aggregate(
                mode="fixed_node_budget",
                planner="determinized_astar",
                runs=node_runs,
                reference=reference,
            ),
            **common,
        },
        {
            **_aggregate(
                mode="fixed_wall_clock_budget",
                planner="determinized_astar",
                runs=time_runs,
                reference=reference,
            ),
            **common,
        },
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    axes[0].bar(
        ["exact\nreference", "A*\nnode-matched"],
        [1000 * float(row["mean_seconds"]) for row in rows[:2]],
        color=["#4c78a8", "#f58518"],
    )
    axes[0].set_ylabel("Wall-clock time (ms)")
    axes[0].set_title("Runtime at reference node budget")
    axes[1].bar(
        ["exact\nreference", "A*\ntime-matched"],
        [float(rows[0]["mean_expanded_nodes"]), float(rows[2]["mean_expanded_nodes"])],
        color=["#4c78a8", "#54a24b"],
    )
    axes[1].set_ylabel("Expanded nodes")
    axes[1].set_title("Nodes at reference time budget")
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
