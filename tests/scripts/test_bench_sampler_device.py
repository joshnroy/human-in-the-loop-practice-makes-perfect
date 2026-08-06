"""Tests for the CPU-vs-GPU sampler benchmark driver.

The benchmark's conclusion rests on two things being true of the harness itself, and
both are silent failures if they are not:

1. **The synthetic data must contain both classes.** `MlpBinaryClassifier.fit` takes a
   single-class shortcut that returns *without building a net at all*, so a generator
   that ever produces an all-one-class label vector would time an empty function and
   report a fit that never happened.
2. **The GPU arm must be the same computation as the CPU arm.** If the subclass
   silently trains something else, any speed comparison is meaningless.
"""

import numpy as np
import pytest
import torch

from hitl_pmp.methods.practice_makes_perfect.wrapped_sampler import MlpBinaryClassifier
from scripts.bench_sampler_device import CudaMlpBinaryClassifier, SamplerDeviceBench


@pytest.mark.parametrize("n", [1, 2, 3, 8, 16, 64, 513])
def test_generated_data_always_contains_both_classes(*, n):
    """A single-class draw would silently time the shortcut instead of a real fit."""
    x_data, y_data = SamplerDeviceBench.make_data(n=n, dim=12, seed=7)
    if n == 1:
        # Degenerate by construction: one row cannot hold two classes. The grid never
        # uses it, but the generator must not pretend otherwise.
        assert y_data.sum() == 1
        return
    assert 0 < y_data.sum() < n, f"n={n} produced a single-class label vector"


def test_generated_rows_have_the_real_layout():
    """The real classifier input row is `[1.0 bias] + features`, and the leading
    constant column is what makes `_normalize_data`'s clip-at-1 branch fire."""
    x_data, y_data = SamplerDeviceBench.make_data(n=32, dim=12, seed=0)
    assert x_data.shape == (32, 12)
    assert y_data.shape == (32,)
    assert np.array_equal(x_data[:, 0], np.ones(32))


def test_generated_data_is_seed_determined():
    first = SamplerDeviceBench.make_data(n=16, dim=11, seed=3)
    second = SamplerDeviceBench.make_data(n=16, dim=11, seed=3)
    other = SamplerDeviceBench.make_data(n=16, dim=11, seed=4)
    assert np.array_equal(first[0], second[0])
    assert not np.array_equal(first[0], other[0])


def test_cpu_timing_actually_fits_and_returns_positive_time():
    seconds = SamplerDeviceBench.time_one(device="cpu", n=16, dim=12, iters=3, seed=1)
    assert seconds > 0


def test_cuda_subclass_trains_the_same_net_as_the_cpu_original():
    """Same seed, same data, same iteration count -- the two arms must agree on the
    weights to within float tolerance, or the comparison is between two different
    computations. Skipped where there is no GPU (CI has none)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    x_data, y_data = SamplerDeviceBench.make_data(n=32, dim=12, seed=5)
    cpu = MlpBinaryClassifier(seed=5, max_train_iters=50, n_iter_no_change=10**9)
    gpu = CudaMlpBinaryClassifier(seed=5, max_train_iters=50, n_iter_no_change=10**9)
    cpu.fit(x_data=x_data, y_data=y_data)
    gpu.fit(x_data=x_data, y_data=y_data)
    cpu_probabilities = cpu.predict_proba(x_data=x_data)
    gpu_probabilities = gpu.predict_proba(x_data=x_data)
    assert np.allclose(cpu_probabilities, gpu_probabilities, atol=1e-4)


def test_the_shipped_class_is_used_unmodified_for_the_cpu_arm():
    """The CPU number must come from the class the project actually runs, not a copy."""
    assert SamplerDeviceBench.time_one.__module__ == "scripts.bench_sampler_device"
    assert issubclass(CudaMlpBinaryClassifier, MlpBinaryClassifier)
    # Exactly the two device-bound methods are overridden. `fit` -- normalization, the
    # single-class shortcut, the balancing branch -- and `_build_net`, the architecture,
    # are inherited, so both arms train the same net on the same normalized data.
    overridden = set(CudaMlpBinaryClassifier.__dict__) & {
        "fit",
        "predict_proba",
        "_train",
        "_build_net",
        "_balance",
    }
    assert overridden == {"_train", "predict_proba"}
