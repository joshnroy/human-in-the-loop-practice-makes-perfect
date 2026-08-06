"""Tests for the fit-share profiler.

The profiler replaces two methods on a class the whole project shares. Two failure
modes are silent and would poison every later measurement in the same process:

1. **Not restoring them** when the wrapped run raises.
2. **Counting a single-class shortcut as a fit.** That branch returns without building
   a net, so pooling it with real fits understates the per-fit cost.
"""

import json

import numpy as np
import pytest

from hitl_pmp.methods.practice_makes_perfect import wrapped_sampler
from scripts.profile_sampler_fit_share import SamplerFitShareProfiler


def test_it_records_each_fit_and_distinguishes_the_single_class_shortcut(*, tmp_path, monkeypatch):
    """Three fits: two real, one all-negative (which takes the shortcut)."""

    def fake_main(*, argv):
        classifier = wrapped_sampler.MlpBinaryClassifier(max_train_iters=2)
        both = (np.array([[1.0, 0.2], [1.0, 0.9]]), np.array([0.0, 1.0]))
        classifier.fit(x_data=both[0], y_data=both[1])
        classifier.fit(x_data=both[0], y_data=both[1])
        classifier.fit(x_data=both[0], y_data=np.array([0.0, 0.0]))
        classifier.predict_proba(x_data=both[0])

    monkeypatch.setattr("scripts.profile_sampler_fit_share.Cli.main", fake_main)
    out = tmp_path / "profile.json"
    record = SamplerFitShareProfiler.run(profile_out=str(out), argv=["--env", "x"])

    assert record["fit_count"] == 3
    assert record["trained_fit_count"] == 2
    assert record["predict_count"] == 1
    assert [entry["took_single_class_shortcut"] for entry in record["fits"]] == [
        False,
        False,
        True,
    ]
    assert record["wall_total_seconds"] >= record["fit_total_seconds"]
    assert json.loads(out.read_text())["fit_count"] == 3


def test_the_patched_methods_are_restored_even_when_the_run_raises(*, tmp_path, monkeypatch):
    """A leaked wrapper would keep timing -- and keep appending to a dead list -- for
    the rest of the process."""
    original_fit = wrapped_sampler.MlpBinaryClassifier.fit
    original_predict = wrapped_sampler.MlpBinaryClassifier.predict_proba

    def exploding_main(*, argv):
        raise RuntimeError("boom")

    monkeypatch.setattr("scripts.profile_sampler_fit_share.Cli.main", exploding_main)
    with pytest.raises(RuntimeError, match="boom"):
        SamplerFitShareProfiler.run(profile_out=str(tmp_path / "p.json"), argv=[])

    assert wrapped_sampler.MlpBinaryClassifier.fit is original_fit
    assert wrapped_sampler.MlpBinaryClassifier.predict_proba is original_predict


def test_cli_flags_are_forwarded_verbatim(*, tmp_path, monkeypatch):
    """The run being measured must be the run the CLI would otherwise have done."""
    seen = {}

    def fake_main(*, argv):
        seen["argv"] = argv

    monkeypatch.setattr("scripts.profile_sampler_fit_share.Cli.main", fake_main)
    SamplerFitShareProfiler.main(
        argv=[
            "--profile-out",
            str(tmp_path / "p.json"),
            "--env",
            "tossingroomsplit",
            "--method",
            "ees",
            "--seed",
            "0",
        ]
    )
    assert seen["argv"] == ["--env", "tossingroomsplit", "--method", "ees", "--seed", "0"]
