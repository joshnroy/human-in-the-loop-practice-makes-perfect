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
        """The bracket in n where `fast` overtakes `slow`, as a human-readable string."""
        by_n: dict[int, dict[str, float]] = {}
        for row in rows:
            by_n.setdefault(int(row["n"]), {})[str(row["device"])] = float(row["median_seconds"])
        paired = sorted(n for n, d in by_n.items() if slow in d and fast in d)
        below = [n for n in paired if by_n[n][slow] < by_n[n][fast]]
        above = [n for n in paired if by_n[n][fast] < by_n[n][slow]]
        if not above:
            return f"no crossover: {slow} is faster at every measured n (max n = {max(paired)})"
        if not below:
            return f"no crossover: {fast} is faster at every measured n (min n = {min(paired)})"
        return f"crossover between n = {max(below)} and n = {min(above)}"

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

        bracket = SamplerDeviceBenchPlot.crossover(rows=rows, slow="cpu", fast="cuda")
        if bracket.startswith("crossover"):
            low = int(bracket.split("n = ")[1].split(" ")[0])
            high = int(bracket.rsplit("n = ", maxsplit=1)[1])
            axes.axvspan(low, high, color="#888888", alpha=0.18)
            axes.annotate(
                f"CPU/GPU crossover\n{low} < n < {high}",
                xy=((low * high) ** 0.5, max(ys)),
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
        parser.add_argument(
            "--title",
            default="Sampler classifier fit(), CPU vs GPU (1000 iterations, input dim 12)",
        )
        args = parser.parse_args(argv)
        rows = SamplerDeviceBenchPlot.load(paths=args.bench_json)
        print(SamplerDeviceBenchPlot.crossover(rows=rows, slow="cpu", fast="cuda"))
        print(SamplerDeviceBenchPlot.crossover(rows=rows, slow="cpu1", fast="cuda"))
        SamplerDeviceBenchPlot.plot(rows=rows, out_path=args.out, title=args.title)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    SamplerDeviceBenchPlot.main()
