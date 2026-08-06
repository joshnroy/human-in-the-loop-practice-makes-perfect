"""Post-run analysis of the thread sweep `scripts/bench_sampler_device.py --threads N`
produces: sampler `fit()` cost against torch's intra-op thread count, at fixed dataset
size, with the machine load each point was measured at.

Reads recorded JSON only -- it never times anything and never drives a `Method`, per
CLAUDE.md's analysis/ convention.

The load axis is not decoration. The penalty for using more than one thread is
contention-dependent, so a bare thread-vs-seconds curve would read as a property of
torch when it is really a property of torch *on a busy box*. Plotting both makes the
confound visible instead of hiding it.

The shipped default is drawn as its own marker rather than as the `24` point, because
they are different things: not calling `set_num_threads` leaves torch free to decide
per operation, and at a small dataset it decides not to parallelise at all.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class SamplerThreadSweep:
    """Static-method container -- no state between calls."""

    @staticmethod
    def load(*, path: Path) -> dict[str, object]:
        with open(path) as handle:
            return json.load(handle)

    @staticmethod
    def split(*, rows: list[dict[str, object]]) -> tuple[list[dict], list[dict]]:
        """`(explicit thread settings, the shipped default)`.

        `threads == -1` records "set_num_threads was not called", which is a different
        condition from any explicit value and must never be plotted on the numeric
        axis as though it were one.
        """
        explicit = sorted(
            (row for row in rows if int(row["threads"]) >= 0), key=lambda row: int(row["threads"])
        )
        default = [row for row in rows if int(row["threads"]) < 0]
        return explicit, default

    @staticmethod
    def summary(*, rows: list[dict[str, object]]) -> str:
        explicit, default = SamplerThreadSweep.split(rows=rows)
        parts = [f"{int(row['threads'])}t {float(row['median_seconds']):.3f}s" for row in explicit]
        for row in default:
            parts.append(f"default {float(row['median_seconds']):.3f}s")
        return "; ".join(parts)

    @staticmethod
    def plot(*, rows: list[dict[str, object]], out_path: Path, title: str) -> None:
        explicit, default = SamplerThreadSweep.split(rows=rows)
        figure, axes = plt.subplots(figsize=(8.5, 5.0))
        xs = [int(row["threads"]) for row in explicit]
        ys = [float(row["median_seconds"]) for row in explicit]
        axes.plot(
            xs, ys, marker="o", color="#c1272d", linewidth=2, label="explicit set_num_threads(N)"
        )
        for row in default:
            axes.axhline(
                float(row["median_seconds"]),
                color="#1b7837",
                linestyle="--",
                linewidth=2,
                label=f"shipped default, no call ({float(row['median_seconds']):.3f}s)",
            )
        axes.set_xscale("log", base=2)
        axes.set_yscale("log")
        axes.set_xticks(xs)
        axes.set_xticklabels([str(x) for x in xs])
        axes.set_xlabel("torch intra-op threads")
        axes.set_ylabel("median fit() wall time, seconds (log scale)")
        axes.grid(alpha=0.3, which="both")

        loads = axes.twinx()
        loads.plot(
            xs,
            [float(row["load_average_1min"]) for row in explicit],
            marker="s",
            color="#888888",
            linewidth=1.2,
            linestyle=":",
            label="1-min load average when measured",
        )
        loads.set_ylabel("1-minute load average (24 cores)", color="#666666")
        loads.set_ylim(bottom=0)

        handles, labels = axes.get_legend_handles_labels()
        extra_handles, extra_labels = loads.get_legend_handles_labels()
        axes.legend(handles + extra_handles, labels + extra_labels, loc="upper left", fontsize=9)
        axes.set_title(title)
        figure.tight_layout()
        figure.savefig(out_path, dpi=160)
        plt.close(figure)

    @staticmethod
    def main(*, argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--threads-json", type=Path, required=True)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument(
            "--title",
            default="Sampler fit() against torch thread count (n = 16, dim 12, 1000 iterations)",
        )
        args = parser.parse_args(argv)
        payload = SamplerThreadSweep.load(path=args.threads_json)
        rows = payload["rows"]
        print(SamplerThreadSweep.summary(rows=rows))
        SamplerThreadSweep.plot(rows=rows, out_path=args.out, title=args.title)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    SamplerThreadSweep.main()
