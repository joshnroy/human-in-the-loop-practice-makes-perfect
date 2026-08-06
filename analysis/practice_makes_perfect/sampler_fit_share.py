"""Post-run analysis of `scripts/profile_sampler_fit_share.py`'s JSON: where a real
run's wall clock goes, and what each individual sampler refit cost.

Reads recorded JSON only -- it never runs a `Method`, per CLAUDE.md's analysis/
convention.

Counts are reported as `x/y` seconds throughout rather than as a percentage: the
denominator is one run, and a bare "82%" would hide that.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class SamplerFitShare:
    """Static-method container -- no state between calls."""

    @staticmethod
    def load(*, path: Path) -> dict[str, object]:
        with open(path) as handle:
            return json.load(handle)

    @staticmethod
    def summary(*, record: dict[str, object]) -> str:
        wall = float(record["wall_total_seconds"])
        fit = float(record["fit_total_seconds"])
        trained = [e for e in record["fits"] if not e["took_single_class_shortcut"]]
        rows = [int(e["n_rows"]) for e in trained] or [0]
        return (
            f"fit {fit:.1f}s / wall {wall:.1f}s over "
            f"{len(trained)}/{record['fit_count']} fits that actually trained; "
            f"training rows min {min(rows)} median {sorted(rows)[len(rows) // 2]} "
            f"max {max(rows)}"
        )

    @staticmethod
    def plot(*, record: dict[str, object], out_path: Path, title: str) -> None:
        wall = float(record["wall_total_seconds"])
        fit = float(record["fit_total_seconds"])
        figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6))

        left.barh([0], [fit], color="#c1272d", label=f"sampler fit(): {fit:.1f}s")
        left.barh(
            [0],
            [wall - fit],
            left=[fit],
            color="#bbbbbb",
            label=f"everything else: {wall - fit:.1f}s",
        )
        left.set_yticks([])
        left.set_xlabel("wall clock, seconds")
        left.set_title(f"Where one run's {wall:.0f}s goes\nfit = {fit:.1f}/{wall:.1f} seconds")
        left.legend(loc="lower right", fontsize=9)

        trained = [e for e in record["fits"] if not e["took_single_class_shortcut"]]
        shortcut = [e for e in record["fits"] if e["took_single_class_shortcut"]]
        right.scatter(
            [e["n_rows"] for e in trained],
            [e["seconds"] for e in trained],
            color="#c1272d",
            s=26,
            label=f"trained ({len(trained)}/{len(record['fits'])})",
        )
        if shortcut:
            right.scatter(
                [e["n_rows"] for e in shortcut],
                [e["seconds"] for e in shortcut],
                color="#2b6cb0",
                marker="x",
                s=36,
                label=f"single-class shortcut ({len(shortcut)}/{len(record['fits'])})",
            )
        right.set_xlabel("training rows n at the time of the refit")
        right.set_ylabel("that refit's wall time, seconds")
        right.set_title("Every sampler refit in the run")
        right.grid(alpha=0.3)
        right.legend(fontsize=9)

        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(out_path, dpi=160)
        plt.close(figure)

    @staticmethod
    def main(*, argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--profile-json", type=Path, required=True)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--title", default="EES on Tossing Room (split), seed 0, 25 cycles")
        args = parser.parse_args(argv)
        record = SamplerFitShare.load(path=args.profile_json)
        print(SamplerFitShare.summary(record=record))
        SamplerFitShare.plot(record=record, out_path=args.out, title=args.title)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    SamplerFitShare.main()
