"""Time `MlpBinaryClassifier.fit` on CPU against the same fit on a CUDA GPU.

Driver only: it writes a JSON record of median wall times over a
(device x dataset size x input dim x iteration count) grid.
`analysis/practice_makes_perfect/sampler_device_bench.py` reads that back and plots
it. Nothing here runs a `Method` or an `Environment`; it exercises one class.

Four things this is careful about, each of which is a way the benchmark gets
published wrong:

1. **The CPU arm is the shipped class, unmodified.** The GPU arm is
   `CudaMlpBinaryClassifier` below, which overrides `_train` and `predict_proba` to
   place the net and its inputs on the device -- same architecture, same optimizer,
   same full-batch BCE, same best-loss checkpointing, and
   `tests/scripts/test_bench_sampler_device.py` pins that the two arms agree on the
   fitted probabilities.
2. **`torch.cuda.synchronize()` before stopping any GPU timer.** CUDA launches are
   asynchronous, so an unsynchronised timer measures how fast Python can enqueue
   work rather than how fast the work runs.
3. **Early stopping is disabled in both arms** (`n_iter_no_change = 10**9`), so every
   run executes exactly `max_train_iters + 1` iterations. Iteration count is
   results-affecting (docs/experiment-logs/2026-08-03-ballring-iters.md), so it is
   held fixed inside every comparison and never varied against device; letting a
   device-dependent early stop fire would confound the two.
4. **`torch.set_num_threads` is not called** unless `--threads` is given, because
   calling it is not a no-op even when passed the value `get_num_threads()` already
   reports. Measured on this 24-core box, an explicit `set_num_threads(24)` costs
   12.0 s for a fit that costs 0.32 s when the call is omitted -- the explicit call
   eagerly creates a spin-waiting OpenMP pool. The shipped code never calls it, so
   neither does the default here.

Both classes deliberately duplicate the training loop rather than refactoring
`wrapped_sampler.py` to take a device: the whole point of the measurement was to
decide whether that abstraction is worth carrying, and the answer was no.
"""

import argparse
import copy
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn

from hitl_pmp.methods.practice_makes_perfect.wrapped_sampler import MlpBinaryClassifier


class CudaMlpBinaryClassifier(MlpBinaryClassifier):
    """`MlpBinaryClassifier` with the training loop on a CUDA device.

    A benchmark-only subclass, kept here rather than in `src/` on purpose: it exists
    to be measured against the CPU original, not to be used by any `Method`.

    Note that it must override **two** methods, not one. Once `_train` leaves the net
    on the device, the inherited `predict_proba` -- which builds its input with
    `torch.from_numpy`, i.e. on the CPU -- raises `Expected all tensors to be on the
    same device`. That is the cheapest available evidence that "put the sampler on the
    GPU" is not a one-line change: the device has to be threaded through the scoring
    path too, and scoring runs on the hot path (100 candidates per decision) where
    `fit` runs once per skill per cycle.
    """

    def predict_proba(self, *, x_data: np.ndarray) -> np.ndarray:
        if self._single_class_prediction is not None:
            return np.full(x_data.shape[0], self._single_class_prediction, dtype=np.float64)
        if self._net is None or self._input_shift is None or self._input_scale is None:
            raise RuntimeError("CudaMlpBinaryClassifier.predict_proba called before fit.")
        normalized = (x_data - self._input_shift) / self._input_scale
        tensor_x = torch.from_numpy(np.asarray(normalized, dtype=np.float32)).to("cuda")
        with torch.no_grad():
            probabilities = self._net(tensor_x).squeeze(dim=-1)
        return probabilities.detach().cpu().numpy().astype(np.float64)

    def _train(self, *, x_data: np.ndarray, y_data: np.ndarray) -> None:
        device = torch.device("cuda")
        torch.manual_seed(self.seed)
        tensor_x = torch.from_numpy(np.asarray(x_data, dtype=np.float32)).to(device)
        tensor_y = torch.from_numpy(np.asarray(y_data, dtype=np.float32)).to(device)
        loss_fn = nn.BCELoss()
        best_overall_loss = math.inf
        best_overall_state: dict[str, torch.Tensor] | None = None
        for try_index in range(self.n_reinitialize_tries):
            torch.manual_seed(self.seed + try_index)
            net = self._build_net(input_dim=x_data.shape[1]).to(device)
            optimizer = torch.optim.Adam(
                net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
            net.train()
            best_loss = math.inf
            best_iteration = 0
            best_state: dict[str, torch.Tensor] = copy.deepcopy(net.state_dict())
            for iteration in range(self.max_train_iters + 1):
                predictions = net(tensor_x).squeeze(dim=-1)
                loss = loss_fn(predictions, tensor_y)
                # `.item()` forces a device synchronization every iteration. It is
                # kept because the shipped loop does exactly this to decide whether
                # the current weights are the best so far -- removing it would be a
                # different algorithm, not a faster device.
                loss_value = loss.item()
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_iteration = iteration
                    best_state = copy.deepcopy(net.state_dict())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if iteration - best_iteration > self.n_iter_no_change:
                    break
            if best_loss < best_overall_loss:
                best_overall_loss = best_loss
                best_overall_state = best_state
                net.load_state_dict(best_state)
                net.eval()
                self._net = net
            if best_overall_loss < 1:
                break
        assert best_overall_state is not None


class SamplerDeviceBench:
    """Static-method container per this repo's convention -- no state between calls."""

    @staticmethod
    def make_data(*, n: int, dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """A synthetic training set with the real row layout: a leading 1.0 bias
        column, then features. Both classes are always present, or `fit` takes the
        single-class shortcut and never trains at all -- and `time_one` would then
        report ~2e-5 s for a fit that never happened, which is worse than an error.
        A single row cannot hold two classes, so n < 2 is refused rather than served
        degenerately."""
        if n < 2:
            raise ValueError(f"n must be at least 2 for both classes to be present, got {n}.")
        rng = np.random.default_rng(seed)
        x_data = rng.uniform(size=(n, dim))
        x_data[:, 0] = 1.0
        y_data = np.zeros(n, dtype=np.float64)
        y_data[rng.permutation(n)[: max(1, n // 3)]] = 1.0
        return x_data, y_data

    @staticmethod
    def time_one(*, device: str, n: int, dim: int, iters: int, seed: int) -> float:
        x_data, y_data = SamplerDeviceBench.make_data(n=n, dim=dim, seed=seed)
        cls = CudaMlpBinaryClassifier if device == "cuda" else MlpBinaryClassifier
        classifier = cls(seed=seed, max_train_iters=iters, n_iter_no_change=10**9)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        classifier.fit(x_data=x_data, y_data=y_data)
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - start

    @staticmethod
    def run(*, args: argparse.Namespace) -> dict[str, object]:
        for device in args.devices:
            for _ in range(args.warmup_reps):
                SamplerDeviceBench.time_one(device=device, n=64, dim=12, iters=50, seed=999)
        rows: list[dict[str, object]] = []
        for iters in args.iters:
            for dim in args.dims:
                for n in args.ns:
                    for device in args.devices:
                        samples = [
                            SamplerDeviceBench.time_one(
                                device=device, n=n, dim=dim, iters=iters, seed=100 + rep
                            )
                            for rep in range(args.reps)
                        ]
                        rows.append({
                            "device": device,
                            "n": n,
                            "dim": dim,
                            "max_train_iters": iters,
                            "median_seconds": statistics.median(samples),
                            "min_seconds": min(samples),
                            "max_seconds": max(samples),
                            "reps": args.reps,
                            "samples": samples,
                        })
                        print(
                            f"iters={iters} dim={dim} n={n:6d} {device:4s} "
                            f"median={statistics.median(samples):.4f}s",
                            flush=True,
                        )
        return {
            "torch_version": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_num_threads": torch.get_num_threads(),
            "called_set_num_threads": args.threads >= 0,
            # The resolved namespace, so `--ns`/`--reps`/`--devices` stay recoverable
            # from the file, and the machine-wide load the timings were taken against
            # -- this box is shared, and a contention caveat the data cannot
            # substantiate is not a caveat. Same reasoning as config_snapshot.json and
            # timing.json; see CLAUDE.md.
            "resolved_args": {key: value for key, value in vars(args).items() if key != "out"},
            "load_average_1min": os.getloadavg()[0],
            "cpu_count": os.cpu_count(),
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "rows": rows,
        }

    @staticmethod
    def cuda_init(*, args: argparse.Namespace) -> dict[str, object]:
        """CUDA context creation and first-cuBLAS-matmul cost.

        Meaningful **only in a fresh process**: both are lazy one-time initialisations,
        so a second measurement in the same process reads zero. That is why this is a
        mode of its own rather than a row in the grid, and why the caller is expected
        to invoke it once per process.
        """
        start = time.perf_counter()
        tensor = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        context_seconds = time.perf_counter() - start
        start = time.perf_counter()
        _ = tensor.new_ones(12, 32) @ tensor.new_ones(32, 32)
        torch.cuda.synchronize()
        cublas_seconds = time.perf_counter() - start
        return {
            "torch_version": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(0),
            "context_seconds": context_seconds,
            "cublas_seconds": cublas_seconds,
            "resolved_args": {key: value for key, value in vars(args).items() if key != "out"},
        }

    @staticmethod
    def bitwise(*, args: argparse.Namespace) -> dict[str, object]:
        """Are CPU-fitted and GPU-fitted classifier scores bit-identical?

        This is the reproducibility question, not a speed one, and it is decided
        separately from any timing: `stats.json` byte-stability is how this project
        verifies a change did not alter results, so a device that cannot reproduce the
        CPU path's scores exactly is disqualified whatever it costs.
        """
        exact = 0
        worst = 0.0
        for seed in range(args.bitwise_seeds):
            x_data, y_data = SamplerDeviceBench.make_data(n=args.bitwise_n, dim=12, seed=seed)
            cpu = MlpBinaryClassifier(
                seed=seed, max_train_iters=args.bitwise_iters, n_iter_no_change=10**9
            )
            gpu = CudaMlpBinaryClassifier(
                seed=seed, max_train_iters=args.bitwise_iters, n_iter_no_change=10**9
            )
            cpu.fit(x_data=x_data, y_data=y_data)
            gpu.fit(x_data=x_data, y_data=y_data)
            cpu_probabilities = cpu.predict_proba(x_data=x_data)
            gpu_probabilities = gpu.predict_proba(x_data=x_data)
            exact += int(np.array_equal(cpu_probabilities, gpu_probabilities))
            worst = max(worst, float(np.max(np.abs(cpu_probabilities - gpu_probabilities))))
        print(
            f"bit-identical seeds: {exact}/{args.bitwise_seeds}; "
            f"worst absolute score difference {worst:.3e}",
            flush=True,
        )
        return {
            "torch_version": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(0),
            "bit_identical_seeds": exact,
            "total_seeds": args.bitwise_seeds,
            "worst_absolute_score_difference": worst,
            "resolved_args": {key: value for key, value in vars(args).items() if key != "out"},
        }

    @staticmethod
    def main(*, argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--out", required=True, help="JSON file to write.")
        parser.add_argument("--reps", type=int, default=5)
        parser.add_argument("--warmup-reps", type=int, default=2)
        parser.add_argument(
            "--ns", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512, 2048]
        )
        parser.add_argument("--dims", type=int, nargs="+", default=[11, 12])
        parser.add_argument("--iters", type=int, nargs="+", default=[1000, 10000])
        parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
        parser.add_argument(
            "--threads",
            type=int,
            default=-1,
            help="torch intra-op threads; -1 (the default) does not call "
            "set_num_threads at all, which is what the shipped code does.",
        )
        parser.add_argument(
            "--mode",
            choices=["grid", "cuda-init", "bitwise"],
            default="grid",
            help="grid: the device x n timing sweep. cuda-init: per-process CUDA "
            "context cost, valid only in a fresh process. bitwise: whether the two "
            "devices produce bit-identical scores.",
        )
        parser.add_argument("--bitwise-seeds", type=int, default=10)
        parser.add_argument("--bitwise-n", type=int, default=24)
        parser.add_argument("--bitwise-iters", type=int, default=1000)
        args = parser.parse_args(argv)
        if args.threads >= 0:
            torch.set_num_threads(args.threads)
        modes = {
            "grid": SamplerDeviceBench.run,
            "cuda-init": SamplerDeviceBench.cuda_init,
            "bitwise": SamplerDeviceBench.bitwise,
        }
        with open(args.out, "w") as handle:
            json.dump(modes[args.mode](args=args), handle, indent=2)


if __name__ == "__main__":
    SamplerDeviceBench.main()
