"""Measure what share of a real run's wall clock is spent inside the sampler's
`MlpBinaryClassifier.fit`, by running the normal CLI with `fit` and `predict_proba`
wrapped in timers.

Why this and not `cProfile`: the question is a single ratio (fit seconds against
total seconds) plus the shape of each fit's training set, and a wrapper answers it
with no per-call profiler overhead distorting a loop that runs 10,000 times per fit.

Why a driver in `scripts/` and not an `analysis/` script: it *runs* a `Method`, which
is exactly what `analysis/` may not do. It is a thin shell around `hitl_pmp.cli.Cli`
-- every flag after `--profile-out` is forwarded to the CLI verbatim, so the run it
measures is the run the CLI would otherwise have done.

The timing goes to `--profile-out`, never into `stats.json`: `stats.json`'s
byte-stability is what verifies a change did not alter results, and a wall-clock
number in it would break that on every run. The wrapped run still writes its own
byte-identical `stats.json` through `--output-dir` as usual.
"""

import argparse
import json
import time

import numpy as np

from hitl_pmp.cli import Cli
from hitl_pmp.methods.practice_makes_perfect import wrapped_sampler


class SamplerFitShareProfiler:
    """Static-method container -- the per-run state lives in the record it returns."""

    @staticmethod
    def run(*, profile_out: str, argv: list[str]) -> dict[str, object]:
        fits: list[dict[str, object]] = []
        predict_seconds: list[float] = []
        original_fit = wrapped_sampler.MlpBinaryClassifier.fit
        original_predict = wrapped_sampler.MlpBinaryClassifier.predict_proba

        def timed_fit(self, *, x_data: np.ndarray, y_data: np.ndarray) -> None:  # noqa: PLR0917
            start = time.perf_counter()
            original_fit(self, x_data=x_data, y_data=y_data)
            fits.append({
                "seconds": time.perf_counter() - start,
                "n_rows": int(x_data.shape[0]),
                "n_cols": int(x_data.shape[1]) if x_data.ndim == 2 else 0,
                "n_positive": int(y_data.sum()),
                "max_train_iters": int(self.max_train_iters),
                # A single-class training set returns without building a net at
                # all, so it is not a fit and must not be averaged in with ones
                # that are.
                "took_single_class_shortcut": self._single_class_prediction is not None,
            })

        def timed_predict(self, *, x_data: np.ndarray) -> np.ndarray:  # noqa: PLR0917
            start = time.perf_counter()
            result = original_predict(self, x_data=x_data)
            predict_seconds.append(time.perf_counter() - start)
            return result

        wrapped_sampler.MlpBinaryClassifier.fit = timed_fit
        wrapped_sampler.MlpBinaryClassifier.predict_proba = timed_predict
        try:
            wall_start = time.perf_counter()
            Cli.main(argv=argv)
            wall_seconds = time.perf_counter() - wall_start
        finally:
            wrapped_sampler.MlpBinaryClassifier.fit = original_fit
            wrapped_sampler.MlpBinaryClassifier.predict_proba = original_predict

        record: dict[str, object] = {
            "wall_total_seconds": wall_seconds,
            "fit_total_seconds": sum(float(entry["seconds"]) for entry in fits),
            "fit_count": len(fits),
            "trained_fit_count": sum(1 for e in fits if not e["took_single_class_shortcut"]),
            "predict_total_seconds": sum(predict_seconds),
            "predict_count": len(predict_seconds),
            "argv": argv,
            "fits": fits,
        }
        with open(profile_out, "w") as handle:
            json.dump(record, handle, indent=2)
        return record

    @staticmethod
    def main(*, argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--profile-out", required=True, help="JSON file to write.")
        args, forwarded = parser.parse_known_args(argv)
        record = SamplerFitShareProfiler.run(profile_out=args.profile_out, argv=forwarded)
        print(
            f"fit {record['fit_total_seconds']:.1f}s / wall "
            f"{record['wall_total_seconds']:.1f}s over {record['fit_count']} fits"
        )


if __name__ == "__main__":
    SamplerFitShareProfiler.main()
