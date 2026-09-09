"""Compare planner runtime with the number of nodes expanded.

This deliberately uses a small, dependency-free synthetic belief-space model so
the traversal algorithms are measured without simulator or policy-evaluation
noise. Each action has two chance outcomes; determinized A* samples one while
expectimax enumerates both.
"""

import argparse
import csv
import statistics
import time
import tracemalloc
from pathlib import Path

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
    """Three actions and two equiprobable, non-merging outcomes per action."""

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
        outcome_bonus = 0.002 if potential_next_environment_state.path.endswith("R") else 0.0
        return BenchmarkBelief(
            value=belief_state.value + 0.01 * (practice_action.index + 1) + outcome_bonus
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


def _summary(*, trace: SearchTrace) -> dict[str, object]:
    return next(event for event in trace.events if event["event"] == "search_summary")


def benchmark(*, horizons: list[int], repeats: int, output: Path) -> None:
    rows: list[dict[str, int | float | str]] = []
    model = BenchmarkModel()
    for horizon in horizons:
        for planner_name in ("expectimax", "determinized_astar"):
            timings: list[float] = []
            node_counts: list[int] = []
            generated_counts: list[int] = []
            max_frontiers: list[int] = []
            max_depths: list[int] = []
            peak_bytes: list[int] = []
            termination_reasons: list[str] = []
            actions: list[int | str] = []
            for repeat in range(repeats):
                trace = SearchTrace()
                if planner_name == "expectimax":
                    planner = ExpectimaxPlanner[
                        BenchmarkState, BenchmarkBelief, BenchmarkTheta, BenchmarkAction
                    ]()
                else:
                    planner = DeterminizedAStarPlanner[
                        BenchmarkState, BenchmarkBelief, BenchmarkTheta, BenchmarkAction
                    ](max_expansions=10**7, seed=repeat)
                tracemalloc.start()
                started = time.perf_counter()
                _, action = planner.solve(
                    environment_state=BenchmarkState(),
                    summed_cost=0.0,
                    belief_state=BenchmarkBelief(),
                    horizon=horizon,
                    model=model,  # type: ignore[arg-type]
                    num_samples=1,
                    trace=trace,
                )
                timings.append(time.perf_counter() - started)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                summary = _summary(trace=trace)
                node_counts.append(int(summary["expanded_nodes"]))
                generated_counts.append(int(summary["generated_nodes"]))
                max_frontiers.append(int(summary.get("max_frontier_size", 0)))
                max_depths.append(int(summary["max_depth_reached"]))
                peak_bytes.append(peak)
                termination_reasons.append(str(summary["termination_reason"]))
                actions.append(getattr(action, "index", "STOP"))
            rows.append({
                "planner": planner_name,
                "horizon": horizon,
                "repeats": repeats,
                "mean_seconds": statistics.mean(timings),
                "stdev_seconds": statistics.stdev(timings) if repeats > 1 else 0.0,
                "mean_expanded_nodes": statistics.mean(node_counts),
                "mean_generated_nodes": statistics.mean(generated_counts),
                "mean_max_frontier": statistics.mean(max_frontiers),
                "mean_effective_depth": statistics.mean(max_depths),
                "mean_peak_kib": statistics.mean(peak_bytes) / 1024,
                "termination_reasons": ",".join(sorted(set(termination_reasons))),
                "selected_first_actions": ",".join(str(action) for action in actions),
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for planner_name, marker in (("expectimax", "o"), ("determinized_astar", "s")):
        planner_rows = [row for row in rows if row["planner"] == planner_name]
        axis.plot(
            [float(row["mean_expanded_nodes"]) for row in planner_rows],
            [1000 * float(row["mean_seconds"]) for row in planner_rows],
            marker=marker,
            label=planner_name.replace("_", " "),
        )
        for row in planner_rows:
            axis.annotate(
                f"H={row['horizon']}",
                (float(row["mean_expanded_nodes"]), 1000 * float(row["mean_seconds"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Expanded nodes (mean)")
    axis.set_ylabel("Wall-clock time (ms, mean)")
    axis.set_title("Belief-space planner traversal sweep")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    figure.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(figure)
    for row in rows:
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-horizon", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark(
        horizons=list(range(1, args.max_horizon + 1)),
        repeats=args.repeats,
        output=args.output,
    )


if __name__ == "__main__":
    main()
