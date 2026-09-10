"""Measure complete expectimax and iteration-bounded determinized A*.

The benchmark never interrupts expectimax. It reports elapsed time, evaluated
nodes, traversed nodes, maximum depth, action agreement, and simple regret so
the two algorithms can be compared at their observed compute costs.
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
    """Three actions, two equiprobable outcomes, and a depth-six value peak."""

    actions = [BenchmarkAction(index=index) for index in range(3)]

    def __init__(self, *, root_action_index: int | None = None) -> None:
        self.root_action_index = root_action_index

    def sample_policy_values_from_belief(
        self, *, belief_state: BenchmarkBelief, num_samples: int
    ) -> np.ndarray:
        return np.full(num_samples, belief_state.value)

    def G(self, *, policy_value: float, summed_cost: float) -> float:
        return policy_value - 0.01 * summed_cost

    def get_valid_actions(self, *, environment_state: BenchmarkState) -> list[BenchmarkAction]:
        if environment_state.path == "" and self.root_action_index is not None:
            return [self.actions[self.root_action_index]]
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


def _run(*, planner: Any, horizon: int, model: BenchmarkModel | None = None) -> dict[str, Any]:
    trace = SearchTrace()
    value, action = planner.solve(
        environment_state=BenchmarkState(),
        summed_cost=0.0,
        belief_state=BenchmarkBelief(),
        horizon=horizon,
        model=BenchmarkModel() if model is None else model,
        num_samples=1,
        trace=trace,
    )
    summary = next(event for event in trace.events if event["event"] == "search_summary")
    return {
        "elapsed_seconds": float(summary["search_elapsed_seconds"]),
        "expanded_nodes": int(summary["expanded_nodes"]),
        "traversed_nodes": int(summary["traversed_nodes"]),
        "evaluated_nodes": int(summary["stop_value_evaluations"]),
        "generated_successors": int(summary["generated_successors"]),
        "max_depth": int(summary["max_depth_reached"]),
        "value": float(value),
        "action": getattr(action, "index", "STOP"),
    }


def benchmark(
    *, reference_horizon: int, iteration_budgets: list[int], repeats: int, output: Path
) -> list[dict[str, Any]]:
    """Write one aggregate row per planner configuration and a comparison plot."""
    exact_runs = [
        _run(
            planner=ExpectimaxPlanner[
                BenchmarkState, BenchmarkBelief, BenchmarkTheta, BenchmarkAction
            ](),
            horizon=reference_horizon,
        )
        for _ in range(repeats)
    ]
    action_values = {
        action.index: _run(
            planner=ExpectimaxPlanner[
                BenchmarkState, BenchmarkBelief, BenchmarkTheta, BenchmarkAction
            ](),
            horizon=reference_horizon,
            model=BenchmarkModel(root_action_index=action.index),
        )["value"]
        for action in BenchmarkModel.actions
    }
    action_values["STOP"] = 0.0
    reference_action = exact_runs[0]["action"]
    best_action_value = max(action_values.values())
    configurations: list[tuple[str, int | None, list[dict[str, Any]]]] = [
        ("expectimax", None, exact_runs)
    ]
    for iterations in iteration_budgets:
        configurations.append((
            "determinized_astar",
            iterations,
            [
                _run(
                    planner=DeterminizedAStarPlanner[
                        BenchmarkState,
                        BenchmarkBelief,
                        BenchmarkTheta,
                        BenchmarkAction,
                    ](max_iterations=iterations, seed=repeat),
                    horizon=reference_horizon,
                )
                for repeat in range(repeats)
            ],
        ))

    rows: list[dict[str, Any]] = []
    for planner, iterations, runs in configurations:
        rows.append({
            "planner": planner,
            "max_iterations": "" if iterations is None else iterations,
            "repeats": repeats,
            "mean_elapsed_seconds": statistics.mean(run["elapsed_seconds"] for run in runs),
            "mean_evaluated_nodes": statistics.mean(run["evaluated_nodes"] for run in runs),
            "mean_expanded_nodes": statistics.mean(run["expanded_nodes"] for run in runs),
            "mean_traversed_nodes": statistics.mean(run["traversed_nodes"] for run in runs),
            "mean_generated_successors": statistics.mean(
                run["generated_successors"] for run in runs
            ),
            "mean_max_depth": statistics.mean(run["max_depth"] for run in runs),
            "action_agreement_rate": statistics.mean(
                run["action"] == reference_action for run in runs
            ),
            "mean_simple_regret": statistics.mean(
                best_action_value - action_values[run["action"]] for run in runs
            ),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for row in rows:
        label = str(row["planner"])
        if row["max_iterations"] != "":
            label += f" ({row['max_iterations']} iterations)"
        axes[0].scatter(
            row["mean_evaluated_nodes"], 1000 * float(row["mean_elapsed_seconds"]), label=label
        )
        axes[1].scatter(
            1000 * float(row["mean_elapsed_seconds"]), row["mean_simple_regret"], label=label
        )
    axes[0].set(xlabel="Stop-value evaluations", ylabel="Wall-clock time (ms)")
    axes[1].set(xlabel="Wall-clock time (ms)", ylabel="Simple regret")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    figure.suptitle("Complete expectimax vs iteration-bounded determinized A*")
    figure.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(figure)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-horizon", type=int, default=6)
    parser.add_argument("--astar-iterations", type=int, nargs="+", default=[10, 30, 100, 300])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for row in benchmark(
        reference_horizon=args.reference_horizon,
        iteration_budgets=args.astar_iterations,
        repeats=args.repeats,
        output=args.output,
    ):
        print(row)


if __name__ == "__main__":
    main()
