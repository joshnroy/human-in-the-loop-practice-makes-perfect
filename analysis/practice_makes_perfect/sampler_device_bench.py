"""Post-run analysis of `scripts/bench_sampler_device.py`'s JSON: plots sampler
`fit()` wall time against training-set size for each device, log-log, and marks any
crossover -- the n above which the GPU arm is faster.

Reads recorded JSON only. It never times anything itself and never drives a
`Method`, per CLAUDE.md's analysis/ convention; re-running it on the same file
reproduces the same figure.

A "crossover" here is bracketed, not interpolated: it is reported as the interval
between the largest n at which CPU is faster and the smallest n at which GPU is,
because the grid is coarse and a log-linear interpolation between two decades of n
would invent precision the measurement does not have.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEVICE_LABELS = {
    "cpu": "CPU (shipped: no set_num_threads call)",
    "cpu1": "CPU, torch.set_num_threads(1)",
    "cuda": "CUDA (RTX 5090)",
}
DEVICE_COLORS = {"cpu": "#c1272d", "cpu1": "#1b7837", "cuda": "#2b6cb0"}


class SamplerDeviceBenchPlot:
    """Static-method container -- no state between calls."""

    @staticmethod
    def load(*, paths: list[Path]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for path in paths:
            with open(path) as handle:
                payload = json.load(handle)
            for row in payload["rows"]:
                # A file produced with --threads 1 is a distinct arm, not another
                # sample of the default CPU arm; relabel it so the two never merge.
                if row["device"] == "cpu" and payload.get("called_set_num_threads"):
                    row = {**row, "device": "cpu1"}
                rows.append(row)
        return rows

    @staticmethod
    def crossover(*, rows: list[dict[str, object]], slow: str, fast: str) -> str:
        """The bracket in n where `fast` overtakes `slow`, as a human-readable string.

        Three things this refuses to do, because each would state more than the data
        supports:

        - **Compare arms that were not both measured.** An `n` present for only one
          device is dropped, and if that leaves nothing the answer is "not comparable"
          rather than a crash or a comparison against a missing value.
        - **Bracket non-monotone data.** `max(below)`/`min(above)` is a bracket only if
          the sign of `slow - fast` flips exactly once in n. On a coarse grid measured
          on a shared box it may not, and that form would then emit a *reversed*
          interval. So the sign changes are counted, and more than one is reported as
          having no meaningful bracket.
        - **Conflate the two directions.** `fast` overtaking `slow` as n grows and
          `fast` losing its lead as n grows are different findings, and get different
          sentences.
        """
        by_n: dict[int, dict[str, float]] = {}
        for row in rows:
            by_n.setdefault(int(row["n"]), {})[str(row["device"])] = float(row["median_seconds"])
        paired = sorted(n for n, devices in by_n.items() if slow in devices and fast in devices)
        if not paired:
            return f"not comparable: no n was measured on both {slow} and {fast}"
        if all(by_n[n][fast] == by_n[n][slow] for n in paired):
            return f"no crossover: {slow} and {fast} tie at every measured n"
        # True where `fast` wins. An exact tie counts as "fast has not overtaken".
        fast_wins = [by_n[n][fast] < by_n[n][slow] for n in paired]
        if not any(fast_wins):
            return f"no crossover: {slow} is faster at every measured n (max n = {max(paired)})"
        if all(fast_wins):
            return f"no crossover: {fast} is faster at every measured n (min n = {min(paired)})"
        flips = [i for i in range(1, len(fast_wins)) if fast_wins[i] != fast_wins[i - 1]]
        if len(flips) > 1:
            return (
                f"not a single crossover: the faster arm changes {len(flips)} times "
                f"across n = {paired[0]}..{paired[-1]}, so no bracket is meaningful"
            )
        index = flips[0]
        if fast_wins[index]:
            return f"crossover between n = {paired[index - 1]} and n = {paired[index]}"
        return (
            f"reverse crossover between n = {paired[index - 1]} and n = {paired[index]}: "
            f"{fast} is faster only below it"
        )

    @staticmethod
    def crossover_span(
        *, rows: list[dict[str, object]], slow: str, fast: str
    ) -> tuple[int, int] | None:
        """The same bracket as `crossover`, as numbers, or `None` when there is not
        exactly one forward crossover to draw.

        `plot` needs the interval, not the sentence. Returning it separately keeps the
        figure from depending on the wording of a human-readable string -- and keeps a
        reverse crossover, a multi-flip series and an incomparable pair from all being
        shaded as if they were the same thing.
        """
        message = SamplerDeviceBenchPlot.crossover(rows=rows, slow=slow, fast=fast)
        if not message.startswith("crossover between"):
            return None
        by_n: dict[int, dict[str, float]] = {}
        for row in rows:
            by_n.setdefault(int(row["n"]), {})[str(row["device"])] = float(row["median_seconds"])
        paired = sorted(n for n, devices in by_n.items() if slow in devices and fast in devices)
        fast_wins = [by_n[n][fast] < by_n[n][slow] for n in paired]
        index = next(i for i in range(1, len(fast_wins)) if fast_wins[i] != fast_wins[i - 1])
        return paired[index - 1], paired[index]

    @staticmethod
    def one_condition(*, rows: list[dict[str, object]]) -> tuple[int, int]:
        """The single `(dim, max_train_iters)` every row was measured at.

        The driver's *default* grid spans two dims and two iteration counts, which
        would put four different measurements on the same `(device, n)`. Plotting
        those as one series draws a zigzag between points that differ 10x, and
        `crossover`'s per-n lookup would silently keep whichever row came last. So a
        mixed file is a hard error telling the caller to filter, not something to
        average over.
        """
        conditions = {(int(row["dim"]), int(row["max_train_iters"])) for row in rows}
        if len(conditions) != 1:
            raise ValueError(
                "these rows mix measurement conditions "
                f"(dim, max_train_iters) = {sorted(conditions)}; "
                "pass --dim and --iters to select one before plotting."
            )
        return next(iter(conditions))

    @staticmethod
    def plot(*, rows: list[dict[str, object]], out_path: Path, title: str) -> None:
        figure, axes = plt.subplots(figsize=(8.5, 5.5))
        devices = [d for d in ("cpu", "cpu1", "cuda") if any(r["device"] == d for r in rows)]
        for device in devices:
            points = sorted(
                ((int(r["n"]), float(r["median_seconds"])) for r in rows if r["device"] == device),
                key=lambda pair: pair[0],
            )
            lows = [
                min(float(r["min_seconds"]) for r in rows if r["device"] == device and r["n"] == n)
                for n, _ in points
            ]
            highs = [
                max(float(r["max_seconds"]) for r in rows if r["device"] == device and r["n"] == n)
                for n, _ in points
            ]
            xs = [n for n, _ in points]
            ys = [seconds for _, seconds in points]
            axes.plot(
                xs,
                ys,
                marker="o",
                color=DEVICE_COLORS[device],
                label=DEVICE_LABELS[device],
                linewidth=2,
            )
            axes.fill_between(xs, lows, highs, color=DEVICE_COLORS[device], alpha=0.15)

        # The numeric bracket, not a re-parse of the sentence `crossover` returns.
        span = SamplerDeviceBenchPlot.crossover_span(rows=rows, slow="cpu", fast="cuda")
        if span is not None:
            low, high = span
            axes.axvspan(low, high, color="#888888", alpha=0.18)
            axes.annotate(
                f"CPU/GPU crossover\n{low} < n < {high}",
                # Sit the label just above the CUDA arm, deliberately: the shipped-CPU
                # arm spans two decades above it and anchoring to the overall maximum
                # would push the text off into empty space at the top of the axes.
                xy=(
                    (low * high) ** 0.5,
                    max(float(r["median_seconds"]) for r in rows if r["device"] == "cuda"),
                ),
                ha="center",
                fontsize=9,
                color="#333333",
            )

        axes.set_xscale("log", base=2)
        axes.set_yscale("log")
        axes.set_xlabel("training rows n (log scale)")
        axes.set_ylabel("median fit() wall time, seconds (log scale)")
        axes.set_title(title)
        axes.grid(alpha=0.3, which="both")
        axes.legend(loc="upper left", fontsize=9)
        figure.tight_layout()
        figure.savefig(out_path, dpi=160)
        plt.close(figure)

    @staticmethod
    def main(*, argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--bench-json", type=Path, nargs="+", required=True)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--dim", type=int, help="Keep only rows at this input dim.")
        parser.add_argument("--iters", type=int, help="Keep only rows at this max_train_iters.")
        parser.add_argument(
            "--title",
            default="Sampler classifier fit(), CPU vs GPU (1000 iterations, input dim 12)",
        )
        args = parser.parse_args(argv)
        rows = SamplerDeviceBenchPlot.load(paths=args.bench_json)
        if args.dim is not None:
            rows = [row for row in rows if int(row["dim"]) == args.dim]
        if args.iters is not None:
            rows = [row for row in rows if int(row["max_train_iters"]) == args.iters]
        # Raises on a file mixing dims or iteration counts, rather than plotting four
        # different measurements as one series.
        dim, iters = SamplerDeviceBenchPlot.one_condition(rows=rows)
        print(f"input dim {dim}, {iters} training iterations")
        # Each pair is reported independently, so a file holding only one CPU arm still
        # answers for the arm it has instead of failing on the one it does not.
        for slow in ("cpu", "cpu1"):
            print(SamplerDeviceBenchPlot.crossover(rows=rows, slow=slow, fast="cuda"))
        SamplerDeviceBenchPlot.plot(rows=rows, out_path=args.out, title=args.title)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    SamplerDeviceBenchPlot.main()
