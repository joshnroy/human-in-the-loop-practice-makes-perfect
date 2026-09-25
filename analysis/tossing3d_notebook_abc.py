"""Tossing3D competence-evidence A/B/C: Model A + grid, evidence x cost lambda x seed.

Reads `<results-root>/<evidence>-lambda<lam>/pomdp/<seed>/` trees written by
`scripts/run_sweep.py` and produces the experiment log's figures, a per-run summary
JSON and paired tests across the shared seeds. Post-run only: it drives nothing.

    scripts/with_env.sh python analysis/tossing3d_notebook_abc.py \
        --results-root results/notebook-abc/sweep --output-dir /tmp/abc
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

TOSS = "MoveToTossLocationAndToss"
EVIDENCE = ("all", "non_epsilon", "informed")
LAMBDAS = ("0.0003", "0")
LINESTYLES = {"all": "solid", "non_epsilon": (0, (4, 2)), "informed": (0, (1, 1.5))}
BLUE = "#0072B2"
PAIRS = (("all", "non_epsilon"), ("informed", "non_epsilon"), ("all", "informed"))


class RunSummary:
    """One run's per-cycle series, extracted once from its logs."""

    @staticmethod
    def extract(*, run_dir: Path) -> dict:
        evals = [json.loads(line)["num_solved"] for line in (run_dir / "progress.jsonl").open()]
        run_stats = json.loads((run_dir / "stats.json").read_text())
        transitions = [entry[0] for entry in run_stats["evaluations"]]
        tallies = []
        for cycle in run_stats["practice_outcomes_per_cycle"]:
            toss = cycle.get(TOSS, {})
            attempts, successes = toss.get("num_attempts", 0), toss.get("num_successes", 0)
            random_n, random_s = (
                toss.get("num_random_attempts", 0),
                toss.get("num_random_successes", 0),
            )
            informed_n, informed_s = (
                toss.get("num_informed_attempts", 0),
                toss.get("num_informed_successes", 0),
            )
            tallies.append({
                "attempts": attempts,
                "successes": successes,
                "epsilon_random": [random_s, random_n],
                "informed": [informed_s, informed_n],
                "uninformative": [
                    successes - random_s - informed_s,
                    attempts - random_n - informed_n,
                ],
            })
        clock, last_smoothing, conditioned = [], None, [0, 0]
        with (run_dir / "pomdp_decisions.jsonl").open() as stream:
            for line in stream:
                head = line[:120]
                if '"event": "refit"' in head:
                    clock.append(json.loads(line)["beliefs"][TOSS]["total_training_examples"])
                elif '"event": "smoothing"' in head:
                    last_smoothing = line
                elif '"event": "outcome"' in head and f'"skill": "{TOSS}"' in line[:400]:
                    record = json.loads(line)
                    conditioned[0] += int(bool(record.get("competence_conditioned")))
                    conditioned[1] += 1
        assert last_smoothing is not None, run_dir
        history = json.loads(last_smoothing)["history"][TOSS]
        return {
            "evals": evals,
            "practice_actions": [int(v) for v in np.diff(transitions)],
            "toss": tallies,
            "clock": clock,
            "filtered_competence": [row["filtered_competence"] for row in history],
            "smoothed_competence": [row["smoothed_competence"] for row in history],
            "learning_rate": [row["filtered_learning_rate"] for row in history],
            "toss_outcomes_conditioned": conditioned,
        }


class AbcAnalysis:
    @staticmethod
    def load(*, results_root: Path, cache: Path) -> dict[str, dict[int, dict]]:
        cached = json.loads(cache.read_text()) if cache.exists() else {}
        arms: dict[str, dict[int, dict]] = {}
        for evidence in EVIDENCE:
            for lam in LAMBDAS:
                arm = f"{evidence}-lambda{lam}"
                for run_dir in sorted((results_root / arm / "pomdp").glob("*")):
                    if not (run_dir / "stats.json").exists():
                        continue
                    key = f"{arm}/{run_dir.name}"
                    if key not in cached:
                        cached[key] = RunSummary.extract(run_dir=run_dir)
                    arms.setdefault(arm, {})[int(run_dir.name)] = cached[key]
        cache.write_text(json.dumps(cached))
        return arms

    @staticmethod
    def auc(*, evals: list[int]) -> float:
        return float(np.trapz(evals))

    @staticmethod
    def paired_tests(*, arms: dict[str, dict[int, dict]]) -> list[dict]:
        rows = []
        for lam in LAMBDAS:
            for left, right in PAIRS:
                a, b = arms.get(f"{left}-lambda{lam}", {}), arms.get(f"{right}-lambda{lam}", {})
                seeds = sorted(set(a) & set(b))
                if len(seeds) < 2:
                    continue
                for metric in ("final", "auc"):
                    pick = (
                        (lambda run: run["evals"][-1])
                        if metric == "final"
                        else (lambda run: AbcAnalysis.auc(evals=run["evals"]))
                    )
                    x = np.array([pick(a[s]) for s in seeds], dtype=float)
                    y = np.array([pick(b[s]) for s in seeds], dtype=float)
                    diff = x - y
                    t_p = float(stats.ttest_rel(x, y).pvalue) if np.any(diff != diff[0]) else None
                    w_p = float(stats.wilcoxon(x, y).pvalue) if np.any(diff != 0) else 1.0
                    rows.append({
                        "lambda": lam,
                        "comparison": f"{left} - {right}",
                        "metric": metric,
                        "n": len(seeds),
                        "left": x.tolist(),
                        "right": y.tolist(),
                        "mean_difference": float(diff.mean()),
                        "paired_t_p": t_p,
                        "wilcoxon_p": w_p,
                    })
        return rows

    @staticmethod
    def _grid(*, title: str) -> tuple[plt.Figure, np.ndarray]:
        fig, axes = plt.subplots(
            len(LAMBDAS), len(EVIDENCE), figsize=(13, 6.6), sharex=True, sharey=True
        )
        fig.suptitle(title)
        return fig, axes

    @staticmethod
    def _traces(
        *, ax: plt.Axes, series: list[list[float]], label: str, linestyle: object = "solid"
    ) -> None:
        width = min(len(s) for s in series)
        data = np.array([s[:width] for s in series], dtype=float)
        x = np.arange(width)
        for row in data:
            ax.plot(x, row, color=BLUE, alpha=0.16, linewidth=0.8, linestyle=linestyle)
        ax.plot(
            x,
            data.mean(axis=0),
            color=BLUE,
            linewidth=2.3,
            linestyle=linestyle,
            label=f"{label} — mean, n={len(series)}",
        )
        ax.fill_between(
            x,
            data.mean(axis=0) - data.std(axis=0),
            data.mean(axis=0) + data.std(axis=0),
            color=BLUE,
            alpha=0.08,
            linewidth=0,
        )

    @staticmethod
    def plot_evals(*, arms: dict, output: Path) -> None:
        fig, axes = AbcAnalysis._grid(title="Evaluation tasks solved (of 10) per seed")
        for row, lam in enumerate(LAMBDAS):
            for col, evidence in enumerate(EVIDENCE):
                ax = axes[row, col]
                runs = arms.get(f"{evidence}-lambda{lam}", {})
                if runs:
                    AbcAnalysis._traces(
                        ax=ax,
                        series=[r["evals"] for r in runs.values()],
                        label="solved ± std",
                        linestyle=LINESTYLES[evidence],
                    )
                    ax.legend(fontsize=7, loc="upper left")
                ax.set_title(f"evidence={evidence}, λ={lam}", fontsize=9)
                ax.set_ylim(-0.3, 10.3)
                if row == len(LAMBDAS) - 1:
                    ax.set_xlabel("evaluation (0 = before practice, k = after cycle k)")
                if col == 0:
                    ax.set_ylabel("solved per seed")
        fig.tight_layout()
        fig.savefig(output, dpi=130)
        plt.close(fig)

    @staticmethod
    def plot_overlay(*, arms: dict, output: Path) -> None:
        fig, axes = plt.subplots(1, len(LAMBDAS), figsize=(12, 4), sharey=True)
        for ax, lam in zip(axes, LAMBDAS, strict=True):
            for evidence in EVIDENCE:
                runs = arms.get(f"{evidence}-lambda{lam}", {})
                if not runs:
                    continue
                data = np.array([r["evals"] for r in runs.values()], dtype=float)
                ax.plot(
                    data.mean(axis=0),
                    color=BLUE,
                    linewidth=2.3,
                    linestyle=LINESTYLES[evidence],
                    label=f"{evidence} — mean, n={len(runs)}",
                )
            ax.set_title(f"Evaluation tasks solved (of 10), λ={lam}")
            ax.set_xlabel("evaluation (0 = before practice)")
            ax.set_ylim(-0.3, 10.3)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("solved, mean over seeds")
        fig.tight_layout()
        fig.savefig(output, dpi=130)
        plt.close(fig)

    @staticmethod
    def plot_competence(*, arms: dict, output: Path) -> None:
        fig, axes = AbcAnalysis._grid(
            title="Toss competence: filtered E[C] at cycle end (blue) vs empirical success "
            "that cycle (o all attempts, x informed only)"
        )
        for row, lam in enumerate(LAMBDAS):
            for col, evidence in enumerate(EVIDENCE):
                ax = axes[row, col]
                runs = arms.get(f"{evidence}-lambda{lam}", {})
                if runs:
                    AbcAnalysis._traces(
                        ax=ax,
                        series=[r["filtered_competence"] for r in runs.values()],
                        label="filtered E[C]",
                    )
                    for run in runs.values():
                        rate = [
                            t["successes"] / t["attempts"] if t["attempts"] else np.nan
                            for t in run["toss"][:-1]
                        ]
                        informed = [
                            t["informed"][0] / t["informed"][1] if t["informed"][1] else np.nan
                            for t in run["toss"][:-1]
                        ]
                        ax.plot(rate, "o", mfc="none", mec="black", markersize=3.5, alpha=0.6)
                        ax.plot(informed, "x", color="black", markersize=3, alpha=0.6)
                    ax.legend(fontsize=7, loc="upper left")
                ax.set_title(f"evidence={evidence}, λ={lam}", fontsize=9)
                ax.set_ylim(-0.03, 1.03)
                if row == len(LAMBDAS) - 1:
                    ax.set_xlabel("practice cycle")
                if col == 0:
                    ax.set_ylabel("toss competence")
        fig.tight_layout()
        fig.savefig(output, dpi=130)
        plt.close(fig)

    @staticmethod
    def plot_learning(*, arms: dict, output: Path) -> None:
        fig, axes = plt.subplots(2, len(LAMBDAS), figsize=(12, 7), sharex=True)
        for col, lam in enumerate(LAMBDAS):
            for evidence in EVIDENCE:
                runs = arms.get(f"{evidence}-lambda{lam}", {})
                if not runs:
                    continue
                AbcAnalysis._traces(
                    ax=axes[0, col],
                    series=[r["learning_rate"] for r in runs.values()],
                    label=evidence,
                    linestyle=LINESTYLES[evidence],
                )
                AbcAnalysis._traces(
                    ax=axes[1, col],
                    series=[r["clock"] for r in runs.values()],
                    label=evidence,
                    linestyle=LINESTYLES[evidence],
                )
            axes[0, col].set_yscale("log")
            axes[0, col].set_title(f"Toss learning-rate diagnostic, λ={lam}")
            axes[1, col].set_title(f"Toss training clock (examples at each refit), λ={lam}")
            axes[1, col].set_xlabel("practice cycle")
            for ax in axes[:, col]:
                ax.legend(fontsize=7)
        axes[0, 0].set_ylabel("dE[C]/dm, posterior mean")
        axes[1, 0].set_ylabel("training examples")
        fig.tight_layout()
        fig.savefig(output, dpi=130)
        plt.close(fig)

    @staticmethod
    def plot_attempts(*, arms: dict, output: Path) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.4), gridspec_kw={"width_ratios": [1.6, 1, 1]})
        labels, kinds = [], ("informed", "uninformative", "epsilon_random")
        hatches = {"informed": "", "uninformative": "//", "epsilon_random": ".."}
        for lam in LAMBDAS:
            for evidence in EVIDENCE:
                labels.append(f"{evidence}\nλ={lam}")
        x = np.arange(len(labels))
        bottoms = np.zeros(len(labels))
        for kind in kinds:
            totals = []
            for lam in LAMBDAS:
                for evidence in EVIDENCE:
                    runs = arms.get(f"{evidence}-lambda{lam}", {}).values()
                    s = sum(t[kind][0] for r in runs for t in r["toss"])
                    n = sum(t[kind][1] for r in runs for t in r["toss"])
                    totals.append((s, n))
            n_arr = np.array([n for _, n in totals], dtype=float)
            axes[0].bar(
                x,
                n_arr,
                bottom=bottoms,
                color=BLUE,
                alpha=0.35,
                edgecolor=BLUE,
                hatch=hatches[kind],
                label=kind,
            )
            for xi, (s, n), b in zip(x, totals, bottoms, strict=True):
                if n:
                    axes[0].text(xi, b + n / 2, f"{s}/{n}", ha="center", va="center", fontsize=7)
            bottoms += n_arr
        axes[0].set_xticks(x, labels, fontsize=7)
        axes[0].set_ylabel("toss attempts, summed over seeds")
        axes[0].set_title("Toss attempts by consultation (labels: successes/attempts)")
        axes[0].legend(fontsize=7)
        for ax, lam in zip(axes[1:], LAMBDAS, strict=True):
            for evidence in EVIDENCE:
                runs = arms.get(f"{evidence}-lambda{lam}", {})
                if runs:
                    AbcAnalysis._traces(
                        ax=ax,
                        series=[r["practice_actions"] for r in runs.values()],
                        label=evidence,
                        linestyle=LINESTYLES[evidence],
                    )
            ax.set_title(f"Practice actions per cycle (budget 20), λ={lam}")
            ax.set_xlabel("practice cycle")
            ax.set_ylim(0, 21)
            ax.legend(fontsize=7)
        axes[1].set_ylabel("actions")
        fig.tight_layout()
        fig.savefig(output, dpi=130)
        plt.close(fig)

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--results-root", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--prefix", default="notebook-abc")
        args = parser.parse_args()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        arms = AbcAnalysis.load(
            results_root=args.results_root, cache=args.output_dir / f"{args.prefix}-runs.json"
        )
        out = args.output_dir
        AbcAnalysis.plot_evals(arms=arms, output=out / f"{args.prefix}-evals.png")
        AbcAnalysis.plot_overlay(arms=arms, output=out / f"{args.prefix}-evals-overlay.png")
        AbcAnalysis.plot_competence(arms=arms, output=out / f"{args.prefix}-competence.png")
        AbcAnalysis.plot_learning(arms=arms, output=out / f"{args.prefix}-learning.png")
        AbcAnalysis.plot_attempts(arms=arms, output=out / f"{args.prefix}-attempts.png")
        tests = AbcAnalysis.paired_tests(arms=arms)
        (out / f"{args.prefix}-tests.json").write_text(json.dumps(tests, indent=1))
        for arm, runs in sorted(arms.items()):
            finals = [runs[s]["evals"][-1] for s in sorted(runs)]
            aucs = [AbcAnalysis.auc(evals=runs[s]["evals"]) for s in sorted(runs)]
            print(f"{arm:24s} seeds={sorted(runs)} final={finals} auc={aucs}")
        for row in tests:
            print(
                f"λ={row['lambda']:7s} {row['comparison']:26s} {row['metric']:5s} n={row['n']} "
                f"Δ={row['mean_difference']:+.2f} t_p={row['paired_t_p']} w_p={row['wilcoxon_p']}"
            )


if __name__ == "__main__":
    AbcAnalysis.main()
