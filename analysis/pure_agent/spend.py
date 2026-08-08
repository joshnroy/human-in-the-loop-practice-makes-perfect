"""Post-run analysis of what `--method pure-agent` actually spent: reads the
`agent_calls.jsonl` ledger each run journals beside its `stats.json` and produces the
spend table and the spend figure a report has to carry.

Never runs a simulation and never queries an agent -- see CLAUDE.md's `analysis/`
convention. It only reads `--results-root DIR` laid out as `DIR/<method>/<seed>/`, or a
single `--ledger PATH`.

**Why this is a deliverable rather than bookkeeping.** This baseline makes one network
call per environment step against a weekly subscription allowance with no overflow, and
the standing condition on running it uncapped at all was that the spend be logged. A
number nobody can reproduce from the artifact is not a log, so every figure printed here
comes from the ledger and the ledger alone.

**Every total is a LOWER BOUND.** A call whose `result` message carried no
`total_cost_usd` contributes nothing, and `calls missing a cost` is printed beside every
total so the two are never separable. Costs are **API-equivalent subscription allowance,
not an invoice**: the CLI authenticates against a subscription and reports what the tokens
would have cost, which is the right quantity for comparing arms and the wrong one to call
money owed.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hitl_pmp.methods.pure_agent.types import (  # noqa: E402
    AgentCallKind,
    AgentCallRecord,
    AgentPhase,
    PureAgentLedger,
)

LEDGER_NAME = "agent_calls.jsonl"


class SpendAnalysis:
    """A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    @staticmethod
    def load(*, path: Path) -> PureAgentLedger:
        """One run's ledger. Malformed lines are **skipped and counted**, not fatal: the
        file is journalled line by line during a multi-hour run, so a process killed
        mid-write leaves a truncated last line, and refusing to read a ledger because its
        final line is half-written would throw away every complete record before it."""
        records: list[AgentCallRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(AgentCallRecord.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValueError):
                continue
        return PureAgentLedger(records=records)

    @staticmethod
    def find_ledgers(*, results_root: Path) -> dict[str, Path]:
        """`{"<method>/<seed>": path}` for every run under a results root, found by
        filename glob (the layout `scripts/run_sweep.py` writes and `analysis/` globs)."""
        return {
            f"{path.parent.parent.name}/{path.parent.name}": path
            for path in sorted(results_root.glob(f"*/*/{LEDGER_NAME}"))
        }

    @staticmethod
    def cell_rows(*, ledger: PureAgentLedger) -> list[dict[str, object]]:
        """One row per (phase, kind) cell: how many calls, what they cost, and how the
        per-call cost is distributed.

        Split by kind as well as phase because the two scale with completely different
        things -- `DECISION` with the horizon, `OPENING` with the number of episodes, and
        `DIGEST` with the number of cycles. A single mean over all three answers no
        planning question at all."""
        rows: list[dict[str, object]] = []
        for phase in AgentPhase:
            for kind in AgentCallKind:
                cell = [
                    record
                    for record in ledger.records
                    if record.phase is phase and record.kind is kind
                ]
                if not cell:
                    continue
                costs = [record.total_cost_usd for record in cell if record.total_cost_usd]
                rows.append({
                    "phase": str(phase),
                    "kind": str(kind),
                    "calls": len(cell),
                    "missing_cost": sum(1 for r in cell if r.total_cost_usd is None),
                    "total_usd": sum(costs),
                    "mean_usd": statistics.mean(costs) if costs else 0.0,
                    "median_usd": statistics.median(costs) if costs else 0.0,
                    "max_usd": max(costs) if costs else 0.0,
                    "mean_seconds": statistics.mean(record.seconds for record in cell),
                })
        return rows

    @staticmethod
    def print_report(*, name: str, ledger: PureAgentLedger) -> None:
        """The spend table for one run, as `x/y` counts throughout.

        A percentage never replaces a count here for the standing reason and for one
        specific to this arm: a malformed-decision rate of "2%" over 92 calls and over
        5,460 calls support completely different claims about whether the parser is
        adequate."""
        total = ledger.num_calls()
        print(f"\n=== {name} ===")
        print(f"calls: {total}   decisions: {ledger.num_decisions()}/{total}")
        print(f"malformed decisions: {ledger.num_malformed_decisions()}/{ledger.num_decisions()}")
        print(
            f"total subscription allowance: ${ledger.total_cost_usd():.2f} "
            f"(LOWER BOUND -- {ledger.num_calls_missing_cost()}/{total} calls reported no "
            "cost; API-equivalent, not an invoice)"
        )
        header = (
            f"{'phase':<11}{'kind':<10}{'calls':>7}{'no cost':>9}{'total $':>10}"
            f"{'mean $':>9}{'median $':>10}{'max $':>9}{'mean s':>9}"
        )
        print(header)
        print("-" * len(header))
        for row in SpendAnalysis.cell_rows(ledger=ledger):
            print(
                f"{row['phase']:<11}{row['kind']:<10}{row['calls']:>7}"
                f"{row['missing_cost']:>9}{row['total_usd']:>10.3f}"
                f"{row['mean_usd']:>9.4f}{row['median_usd']:>10.4f}"
                f"{row['max_usd']:>9.4f}{row['mean_seconds']:>9.2f}"
            )
        SpendAnalysis.print_projection(ledger=ledger)

    @staticmethod
    def print_projection(*, ledger: PureAgentLedger) -> None:
        """What a full-size run would cost, projected from THIS run's own per-call costs.

        **Three pools, not two, and the third is the one that surprises.** Practice and
        evaluation obviously differ -- an evaluation episode is ~12 turns and a practice
        period is up to 150. But evaluation *before any practice* and evaluation *after*
        also differ, by a lot: from cycle 1 onward every opening prompt carries the agent's
        practice digest, which sits in the conversation for the whole episode. Measured on
        the 50-step Tossing Room pilot, that took an evaluation decision from ~$0.032 to
        ~$0.12. Since a `--num-cycles 10` run has ten post-practice sweeps and exactly one
        pre-practice sweep, projecting off a pooled evaluation mean understates the bill by
        roughly 4x. So the post-practice mean is used wherever the run has one.

        Every projection is a **FLOOR**, and the label is not hedging. A short practice
        period does not measure the tail of a long one, because the context keeps growing
        within the period; and the digest itself grows over cycles, so late sweeps cost
        more than the first post-practice sweep this is projected from."""
        practice = [
            r.total_cost_usd
            for r in ledger.records
            if r.phase is AgentPhase.PRACTICE and r.kind is AgentCallKind.DECISION
            if r.total_cost_usd
        ]
        evaluation_all = [
            r
            for r in ledger.records
            if r.phase is AgentPhase.EVALUATION and r.kind is AgentCallKind.DECISION
            if r.total_cost_usd
        ]
        # `cycle_index > 0` is exactly "a digest existed when this call was made": the
        # counter advances in `end_cycle`, which is where the digest is written.
        post_practice = [r.total_cost_usd for r in evaluation_all if r.cycle_index > 0]
        untrained = [r.total_cost_usd for r in evaluation_all if r.cycle_index == 0]
        if not (practice and evaluation_all):
            print(
                "projection: not printed -- this run has no priced calls on one of the "
                "two phases, and pooling them would understate the more expensive one."
            )
            return
        practice_mean = statistics.mean(practice)
        evaluation_mean = statistics.mean(post_practice or untrained)
        print(
            f"per-decision means: practice ${practice_mean:.4f} over {len(practice)} calls, "
            f"evaluation ${evaluation_mean:.4f} over "
            f"{len(post_practice or untrained)} calls"
        )
        if post_practice and untrained:
            print(
                f"  evaluation splits: ${statistics.mean(untrained):.4f} over "
                f"{len(untrained)} untrained calls vs "
                f"${statistics.mean(post_practice):.4f} over {len(post_practice)} "
                "post-practice calls -- the digest rides in every conversation from cycle "
                "1 on, and the projection below uses the POST-PRACTICE figure"
            )
        elif not post_practice:
            print(
                f"  WARNING: no post-practice evaluation calls in this run, so the "
                f"projection uses the {len(untrained)} untrained ones and UNDERSTATES "
                "every cycle after the first -- measured at ~4x on the Tossing Room pilot."
            )
        for cycles, steps, tasks, horizon in ((2, 150, 30, 12), (10, 150, 30, 12)):
            practice_calls = cycles * steps
            evaluation_calls = (cycles + 1) * tasks * horizon
            projected = practice_calls * practice_mean + evaluation_calls * evaluation_mean
            print(
                f"  projected {cycles}-cycle run: {practice_calls + evaluation_calls} "
                f"decisions -> ${projected:.0f} (FLOOR: per-call cost grows with "
                "conversation length, and this run's periods were shorter)"
            )

    @staticmethod
    def plot(*, ledgers: dict[str, PureAgentLedger], output_path: Path) -> None:
        """Two panels: per-call cost against call index, and cumulative spend.

        The per-call panel is the one that carries the finding, and it is a scatter rather
        than a mean: the whole question is whether cost is flat or climbs with conversation
        length, and a mean answers it by destroying it. Practice and evaluation are drawn
        apart because they are two different curves -- an evaluation episode resets every
        ~12 calls while a practice period runs to 150 -- and pooling them produces a
        sawtooth that looks like noise."""
        figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for name, ledger in sorted(ledgers.items()):
            costs = [record.total_cost_usd or 0.0 for record in ledger.records]
            indices = range(len(costs))
            # Indexed off `ledger.records` directly rather than zipped with `costs`: the
            # two are the same list walked twice, and a zip would quietly tolerate a
            # length mismatch that cannot happen and would misalign every point if it did.
            practice = [
                (index, costs[index])
                for index, record in enumerate(ledger.records)
                if record.phase is AgentPhase.PRACTICE
            ]
            evaluation = [
                (index, costs[index])
                for index, record in enumerate(ledger.records)
                if record.phase is AgentPhase.EVALUATION
            ]
            top.scatter(
                [i for i, _ in evaluation],
                [c for _, c in evaluation],
                s=6,
                alpha=0.6,
                label=f"{name} evaluation ({len(evaluation)} calls)",
            )
            top.scatter(
                [i for i, _ in practice],
                [c for _, c in practice],
                s=6,
                alpha=0.6,
                marker="^",
                label=f"{name} practice ({len(practice)} calls)",
            )
            cumulative: list[float] = []
            running = 0.0
            for cost in costs:
                running += cost
                cumulative.append(running)
            bottom.plot(indices, cumulative, label=f"{name} (${running:.2f})")
        top.set_ylabel("subscription allowance per call (USD)")
        top.set_title(
            "Per-call cost and cumulative spend, --method pure-agent\n"
            "API-equivalent allowance, not an invoice; totals are lower bounds"
        )
        top.legend(fontsize="x-small")
        bottom.set_xlabel("agent call index within the run")
        bottom.set_ylabel("cumulative allowance (USD)")
        bottom.legend(fontsize="x-small")
        figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        print(f"\nwrote {output_path}")

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument(
            "--results-root",
            type=Path,
            help=f"A sweep root laid out as DIR/<method>/<seed>/{LEDGER_NAME}.",
        )
        source.add_argument(
            "--ledger", type=Path, help=f"A single run's {LEDGER_NAME}, or its directory."
        )
        parser.add_argument(
            "--figure",
            type=Path,
            default=None,
            help="Where to write the spend figure. Omit to print the table only.",
        )
        args = parser.parse_args()
        if args.results_root is not None:
            paths = SpendAnalysis.find_ledgers(results_root=args.results_root)
        else:
            path = args.ledger / LEDGER_NAME if args.ledger.is_dir() else args.ledger
            paths = {path.parent.name: path}
        if not paths:
            print(f"No {LEDGER_NAME} found. Nothing to report.")
            return
        ledgers = {name: SpendAnalysis.load(path=path) for name, path in paths.items()}
        for name, ledger in sorted(ledgers.items()):
            SpendAnalysis.print_report(name=name, ledger=ledger)
        if len(ledgers) > 1:
            grand = PureAgentLedger(
                records=[record for ledger in ledgers.values() for record in ledger.records]
            )
            SpendAnalysis.print_report(name=f"ALL {len(ledgers)} RUNS", ledger=grand)
        if args.figure is not None:
            SpendAnalysis.plot(ledgers=ledgers, output_path=args.figure)


if __name__ == "__main__":
    SpendAnalysis.main()
